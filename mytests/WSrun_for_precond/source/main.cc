#include <deal.II/base/multithread_info.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/lac/vector.h>

#include <benchmark/benchmark_api.h>

#include <iostream>
#include <vector>
#include <cstdio>

// global settings
const unsigned int dim = 2;
const unsigned int degree = 1;

dealii::Triangulation<2> tr;
const dealii::FE_DGQ<dim> fe{degree};
dealii::DoFHandler<dim> dof_handler{tr};
const dealii::MappingQ1<dim> mapping;

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
      std::cout << "initial refine" << std::endl;
    }
  // re-refine for new tria
  if(prev_range_x != state.range_x() && tr_refined)
    {
      tr.refine_global(state.range_x()-prev_range_x);
      dof_handler.distribute_dofs(fe);
      prev_range_x = state.range_x();
      std::cout << "refine again" << std::endl;
    }
      
  while(state.KeepRunning())
    {
      dealii::Vector<double> global_dst;
      global_dst.reinit(dof_handler.n_dofs());
      dealii::Vector<double> global_src;
      global_src.reinit(dof_handler.n_dofs());
      
      global_dst.print(std::cout);
    }
}

BENCHMARK(mw_constrhs)
->Threads(1)
->ArgPair(0,1)
->UseRealTime();

BENCHMARK_MAIN()
