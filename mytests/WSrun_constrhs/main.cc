#include <deal.II/base/multithread_info.h>
#include <deal.II/base/template_constraints.h>
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

#include <benchmark/benchmark_api.h>

#include <iostream>
#include <vector>
#include <cstdio>
#include <functional>

template <int dim>
void cell_worker(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info)
{
  std::vector<double> f;
  f.resize(info.fe_values(0).n_quadrature_points,1.);
  dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(0),
				   info.fe_values(0),
				   f);
}

// global settings
const unsigned int dim = 2;
const unsigned int degree = 1;

dealii::Triangulation<2> tr;
const dealii::FE_DGQ<dim> fe{degree};
dealii::DoFHandler<dim> dof_handler{tr};
const dealii::MappingQ1<dim> mapping;

dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> > dof_info;
const dealii::MeshWorker::LoopControl lctrl{} ;

bool tr_refined = false;
int prev_range_x = 0;

// MAIN
void mw_constrhs (benchmark::State &state)
{
  dealii::MultithreadInfo::set_thread_limit (state.range_y());
  
  // refine of initial tria
  if(!tr_refined)
    {
      dealii::GridGenerator::hyper_cube (tr,0.,1.);
      tr.refine_global(state.range_x());
      dof_handler.distribute_dofs(fe);
      tr_refined = true;
      prev_range_x = state.range_x();

      info_box.initialize_gauss_quadrature(degree+1, 0, 0);
      dealii::UpdateFlags update_flags = dealii::update_quadrature_points | dealii::update_values ;
      info_box.add_update_flags(update_flags, true, false, false, false);
      info_box.initialize(fe, mapping);
      dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> {dof_handler});
      //std::cout << "initial refine" << std::endl;
    }
  // re-refine for new tria
  if(prev_range_x != state.range_x() && tr_refined)
    {
      tr.refine_global(state.range_x()-prev_range_x);
      dof_handler.distribute_dofs(fe);
      prev_range_x = state.range_x();
      dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> {dof_handler});
      //std::cout << "refine again" << std::endl;
    }
      
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

      // ITERATOR begin{dof_handler.begin_active()} ;
      // typename dealii::identity<ITERATOR>::type end{dof_handler.end()};
      // DOFINFO &dinfo = *dof_info ;
      // INFOBOX &info = info_box  ;
      // ASSEMBLER &assembler = rhs_assembler ;

      // Loop over all cells                                                                                                         
#ifdef DEAL_II_MESHWORKER_PARALLEL
      
      dealii::MeshWorker::DoFInfoBox<dim,DOFINFO> dinfo_box(*dof_info);
      
      rhs_assembler.initialize_info(dinfo_box.cell, false);
      for (unsigned int i=0; i<dealii::GeometryInfo<dim>::faces_per_cell; ++i)
        {
          rhs_assembler.initialize_info(dinfo_box.interior[i], true);
          rhs_assembler.initialize_info(dinfo_box.exterior[i], true);
        }
      
      dealii::WorkStream::run(dof_handler.begin_active(), dof_handler.end(),
      			      std::bind(&dealii::MeshWorker::cell_action<INFOBOX,DOFINFO,dim,dim,ITERATOR>,
					std::placeholders::_1,  std::placeholders::_3,  std::placeholders::_2,
      					cell_worker<dim>, nullptr, nullptr, lctrl),
			      std::bind(&dealii::internal::assemble<dim,DOFINFO,ASSEMBLER>, std::placeholders::_1, &rhs_assembler),
			      info_box, dinfo_box) ;
      
#else
      for (ITERATOR cell = begin; cell != end; ++cell)
        {
	  dealii::WorkStream::cell_action<INFOBOX,DOFINFO,dim,spacedim>(cell,
									dinfo_box, info_box,
									ccell_worker, nullptr, nullptr,
									lctrl);
          dinfo_box.assemble(assembler);
        }
#endif
    
      //      rhs.print(std::cout);
    }
}

BENCHMARK(mw_constrhs)
->Threads(1)
->ArgPair(7,1)
//->Apply(CustomArguments)
->UseRealTime();

BENCHMARK_MAIN()
