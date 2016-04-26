#include <deal.II/base/std_cxx11/function.h>
#include <deal.II/base/multithread_info.h>
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

// functions
template <int dim>
static void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info)
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
      //std::cout << "initial refine" << std::endl;
    }
  // re-refine for new tria
  if(prev_range_x != state.range_x() && tr_refined)
    {
      tr.refine_global(state.range_x()-prev_range_x);
      dof_handler.distribute_dofs(fe);
      prev_range_x = state.range_x();
      //std::cout << "refine again" << std::endl;
    }
      
  while(state.KeepRunning())
    {
      dealii::Vector<double> rhs;
      rhs.reinit(dof_handler.n_dofs());

      dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
      info_box.initialize_gauss_quadrature(degree+1,
					   degree+1,
					   degree+1);
      dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
	dealii::update_values | dealii::update_gradients;
      info_box.add_update_flags(update_flags, true, true, true, true);
      info_box.initialize(fe, mapping);
      
      dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler);
      
      dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > rhs_assembler;
      dealii::AnyData data;
      data.add<dealii::Vector<double>* >(&rhs, "RHS");
      rhs_assembler.initialize(data);
      
      dealii::MeshWorker::loop<dim, dim,
			       dealii::MeshWorker::DoFInfo<dim>,
			       typename dealii::MeshWorker::IntegrationInfoBox<dim> >
	( dof_handler.begin_active(), dof_handler.end(),
	  dof_info, info_box,
	  cell<dim>, nullptr, nullptr,
	  rhs_assembler );

      //      rhs.print(std::cout);
    }
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int x = 6; x <= 12 ; x+=2)
    {
      if(x <= 10)
	{
	  int y0 = (x <= 8) ? 1 : 4;
	  for (int y = y0; y <= 4; y++)
	    b->ArgPair(x, y);
	  for (int y = 6; y <= x; y+=2)
	    b->ArgPair(x, y);
	}
      else
	for (int y = 12; y <= 24; y+=4)
	  b->ArgPair(x, y);
    }
  b->ArgPair(10, 12);
}

BENCHMARK(mw_constrhs)
->Threads(1)
->ArgPair(7,1)
//->Apply(CustomArguments)
->UseRealTime();

BENCHMARK_MAIN()
