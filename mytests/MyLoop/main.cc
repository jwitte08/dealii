//dealii
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

//benchmark
#include <benchmark/benchmark_api.h>

//c++
#include <iostream>
#include <vector>
#include <cstdio>

// RHS INTEGRATOR                                                                                                                    
template <int dim,bool ucell=true,bool ubdry=true,bool uface=true>
class RHSIntegrator
{
 public:
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info){}
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
		typename dealii::MeshWorker::IntegrationInfo<dim> &info){}
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
	    dealii::MeshWorker::DoFInfo<dim> &dinfo2,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info2){}
  bool use_cell = ucell;
  bool use_boundary = ubdry;
  bool use_face = uface;
};

template <int dim>
class RHSIntegrator<dim,true,false,false>
{
 public:
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info)
  {
    std::vector<double> f;
    f.resize(info.fe_values(0).n_quadrature_points,1.);
    dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(0),info.fe_values(0),f);
  } const

  bool use_cell = true;
  bool use_boundary = false;
  bool use_face = false;
};

template<int dim, int spacedim, typename ITERATOR, typename DOFINFO, typename INFOBOX, typename ASSEMBLER>
void colored_loop(const std::vector<std::vector<ITERATOR> > colored_iterators,
		  DOFINFO  &dof_info,
		  INFOBOX  &info,
		  const dealii::std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)> &cell_worker,
		  const dealii::std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)> &boundary_worker,
		  const dealii::std_cxx11::function<void (DOFINFO &, DOFINFO &,
						  typename INFOBOX::CellInfo &,
						  typename INFOBOX::CellInfo &)> &face_worker,
		  ASSEMBLER  &assembler,
		  const dealii::MeshWorker::LoopControl &lctrl = dealii::MeshWorker::LoopControl())
{
  dealii::MeshWorker::DoFInfoBox<dim, DOFINFO> dinfo_box(dof_info);

  assembler.initialize_info(dinfo_box.cell, false);
  for (unsigned int i=0; i<dealii::GeometryInfo<dim>::faces_per_cell; ++i)
    {
      assembler.initialize_info(dinfo_box.interior[i], true);
      assembler.initialize_info(dinfo_box.exterior[i], true);
    }

  // Loop over all cells                                                                                                              
#ifdef DEAL_II_MESHWORKER_PARALLEL
  dealii::WorkStream::run(colored_iterators,
			  dealii::std_cxx11::bind(&dealii::MeshWorker::cell_action<INFOBOX, DOFINFO, dim, spacedim, ITERATOR>,
						  dealii::std_cxx11::_1, dealii::std_cxx11::_3, dealii::std_cxx11::_2,
						  cell_worker, boundary_worker, face_worker, lctrl),
			  dealii::std_cxx11::bind(&dealii::internal::assemble<dim,DOFINFO,ASSEMBLER>,
						  dealii::std_cxx11::_1, &assembler),
			  info, dinfo_box);
#else
  // for (ITERATOR cell = begin; cell != end; ++cell)
  //   {
  //     cell_action<INFOBOX,DOFINFO,dim,spacedim>(cell, dinfo_box,
  // 						info, cell_worker,
  // 						boundary_worker, face_worker,
  // 						lctrl);
  //     dinfo_box.assemble(assembler);
  //   }
#endif
}

// MAIN
void mw_constrhs(benchmark::State &state)
{
  dealii::MultithreadInfo::set_thread_limit (state.range_y());  
  const unsigned int dim = 2;
  const unsigned int degree = 1;

  dealii::Triangulation<2> tr;
  const dealii::FE_DGQ<dim> fe{degree};
  dealii::DoFHandler<dim> dof_handler{tr};
  const dealii::MappingQ1<dim> mapping;
  
  RHSIntegrator<dim,true,false,false> rhs_integrator;
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> > dof_info;
  
  dealii::GridGenerator::hyper_cube (tr,0.,1.);
  tr.refine_global(state.range_x());
  dof_handler.distribute_dofs(fe);

  info_box.initialize_gauss_quadrature(degree+1, 0, 0);
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points | dealii::update_values ;
  info_box.add_update_flags(update_flags, true, false, false, false);
  info_box.initialize(fe, mapping);
  dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> {dof_handler});

  while(state.KeepRunning())
    {
      dealii::Vector<double> rhs;
      rhs.reinit(dof_handler.n_dofs());
      
      dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > rhs_assembler;
      dealii::AnyData data;
      data.add<dealii::Vector<double>* >(&rhs, "RHS");
      rhs_assembler.initialize(data);
      
      //template definitions
      typedef dealii::MeshWorker::DoFInfo<dim> DOFINFO ;
      typedef dealii::MeshWorker::IntegrationInfoBox<dim> INFOBOX ;
      typedef dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > ASSEMBLER ;
      typedef typename dealii::DoFHandler<dim>::active_cell_iterator ITERATOR ;
      typedef RHSIntegrator<dim> INTEGRATOR;

      // ITERATOR begin{dof_handler.begin_active()} ;
      // typename dealii::identity<ITERATOR>::type end{dof_handler.end()};
      // DOFINFO &dinfo = *dof_info ;
      // INFOBOX &info = info_box  ;
      // ASSEMBLER &assembler = rhs_assembler ;

      // build_colored_iterators
      ITERATOR begin{dof_handler.begin_active()} ;
      typename dealii::identity<ITERATOR>::type end{dof_handler.end()};

      // only one color
      std::vector<std::vector<ITERATOR> >  all_iterators(1);
      for (ITERATOR p=begin; p!=end; ++p)
	all_iterators[0].push_back (p);

      colored_loop<dim,dim,ITERATOR,DOFINFO,INFOBOX,ASSEMBLER>
	(all_iterators,
	 *dof_info, info_box,
	 dealii::std_cxx11::bind(&RHSIntegrator<dim,true,false,false>::cell,&rhs_integrator,
				 dealii::std_cxx11::_1,dealii::std_cxx11::_2),
	 /*lambda not working with const RHSIntegrator...[&rhs_integrator](DOFINFO& dinfo, typename INFOBOX::CellInfo& info)
	   {rhs_integrator.cell(dinfo,info);},*/
	 nullptr, nullptr,
	 rhs_assembler);
      
      //output
      //rhs.print(std::cout);
      //std::cout << rhs[0] << " " << rhs[rhs.size()-1] << std::endl;
    }
}

BENCHMARK(mw_constrhs)
->Threads(1)
->ArgPair(11,4)
->UseRealTime();

BENCHMARK_MAIN()
