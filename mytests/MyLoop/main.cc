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
#include <deal.II/integrators/laplace.h>
//benchmark
#include <benchmark/benchmark_api.h>

//c++
#include <iostream>
#include <vector>
#include <cstdio>

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

  //  Loop over all cells                                                                                                              
#ifdef DEAL_II_MESHWORKER_PARALLEL
  dealii::WorkStream::run(colored_iterators,
			  dealii::std_cxx11::bind(&dealii::MeshWorker::cell_action<INFOBOX, DOFINFO, dim, spacedim, ITERATOR>,
						  dealii::std_cxx11::_1, dealii::std_cxx11::_3, dealii::std_cxx11::_2,
						  cell_worker, boundary_worker, face_worker, lctrl),
			  dealii::std_cxx11::bind(&dealii::internal::assemble<dim,DOFINFO,ASSEMBLER>,
						  dealii::std_cxx11::_1, &assembler),
			  info, dinfo_box,
			  2,8);
#else
  for (unsigned int color=0; color<colored_iterators.size(); ++color)
    for (typename std::vector<ITERATOR>::const_iterator p = colored_iterators[color].begin();
	 p != colored_iterators[color].end(); ++p)
      {
	dealii::MeshWorker::cell_action<INFOBOX,DOFINFO,dim,spacedim>(*p, dinfo_box, info,
								      cell_worker, boundary_worker, face_worker,
								      lctrl);
	dinfo_box.assemble(assembler);
      }
#endif
}

// MAIN
void mw_constrhs(benchmark::State &state)
{
  dealii::MultithreadInfo::set_thread_limit (state.range_y());  
  const unsigned int dim = 2;
  const unsigned int degree = 1;

 //template definitions
  typedef dealii::MeshWorker::DoFInfo<dim> DOFINFO ;
  typedef dealii::MeshWorker::IntegrationInfoBox<dim> INFOBOX ;
  typedef dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > ASSEMBLER ;
  typedef typename dealii::DoFHandler<dim>::active_cell_iterator ITERATOR ;
  
  dealii::Triangulation<2> tr;
  const dealii::FE_DGQ<dim> fe{degree};
  dealii::DoFHandler<dim> dof_handler{tr};
  const dealii::MappingQ1<dim> mapping;
  
  INFOBOX info_box;
  std::unique_ptr<DOFINFO> dof_info;
  dealii::MeshWorker::LoopControl loop_control;
  loop_control.own_faces = dealii::MeshWorker::LoopControl::one;
  
  dealii::GridGenerator::hyper_cube (tr,0.,1.);
  tr.refine_global(state.range_x());
  dof_handler.distribute_dofs(fe);

  info_box.initialize_gauss_quadrature(degree+1, degree+1, degree+1);
  dealii::UpdateFlags update_flags = dealii::update_JxW_values | dealii::update_quadrature_points 
    | dealii::update_values | dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);

  // build_colored_iterators
  ITERATOR begin{dof_handler.begin_active()} ;
  typename dealii::identity<ITERATOR>::type end{dof_handler.end()};
  
  // only one color
  std::vector<std::vector<ITERATOR> >  all_iterators(1);
  for (ITERATOR p=begin; p!=end; ++p)
    all_iterators[0].push_back(p);
  
  // two colors
  std::vector<std::vector<ITERATOR> >  colored_iterators(2);
  for (ITERATOR p=begin; p!=end; ++p)
    {colored_iterators[0].push_back(p); ++p;}
  for (ITERATOR p=begin; p!=end; ++p)
    {++p; colored_iterators[1].push_back(p);}
  //  std::cout << colored_iterators[0].size() + colored_iterators[1].size() << std::endl;
  
  // std::cout << "n_cells= " << tr.n_active_cells() << std::endl;
  // std::cout << "n_dofs= " << dof_handler.n_dofs() << std::endl;
  while(state.KeepRunning())
    {
      dealii::Vector<double> src;
      src.reinit(dof_handler.n_dofs());
      src = 0.1;
      dealii::AnyData src_data;
      src_data.add<dealii::Vector<double>* >(&src, "src");

      dealii::Vector<double> dst;
      dst.reinit(dof_handler.n_dofs());
      dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > assembler;
      dealii::AnyData dst_data;
      dst_data.add<dealii::Vector<double>* >(&dst, "dst");
      assembler.initialize(dst_data);
      //std::cout << "BEFORE:dst(10)=" << dst[10] << " ,src(10)=" << src[10] << std::endl;

      info_box.initialize(fe, mapping, src_data, src);
      
      dof_info.reset(new DOFINFO{dof_handler});

      colored_loop<dim,dim,ITERATOR,DOFINFO,INFOBOX,ASSEMBLER>
      	(all_iterators,
      // dealii::MeshWorker::loop<dim,dim,DOFINFO,INFOBOX,ASSEMBLER,ITERATOR>
      // 	(dof_handler.begin_active(),dof_handler.end(),
      	 *dof_info, info_box,
	 
	 [](DOFINFO& dinfo, typename INFOBOX::CellInfo& info)
	 { dealii::LocalIntegrators::Laplace::cell_residual
	     (dinfo.vector(0).block(0),info.fe_values(0),info.gradients[0][0]);},
	 
	 [](DOFINFO& dinfo, typename INFOBOX::CellInfo& info)
	 { std::vector<double> bdry_data;
	   bdry_data.resize(dinfo.vector(0).block(0).size(),0.);
	   dealii::LocalIntegrators::Laplace::nitsche_residual
	     (dinfo.vector(0).block(0), info.fe_values(0),info.values[0][0],info.gradients[0][0],bdry_data,
	      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, degree, degree));},
	 
	 [](DOFINFO& dinfo1, DOFINFO& dinfo2,
	    typename INFOBOX::CellInfo& info1, typename INFOBOX::CellInfo& info2)
	 { dealii::LocalIntegrators::Laplace::ip_residual
	     (dinfo1.vector(0).block(0), dinfo2.vector(0).block(0),
	      info1.fe_values(0), info2.fe_values(0),
	      info1.values[0][0], info1.gradients[0][0],
	      info2.values[0][0], info2.gradients[0][0],
	      dealii::LocalIntegrators::Laplace::
	      compute_penalty(dinfo1, dinfo2, degree, degree));},
      	 assembler, loop_control);
      
      //output
      //dst.print(std::cout);
      //std::cout << "AFTER:dst(10)=" << dst[10] << " ,src(10)=" << src[10] << std::endl;
    }
}

BENCHMARK(mw_constrhs)
->Threads(1)
->ArgPair(9,2)
->UseRealTime();

BENCHMARK_MAIN()
