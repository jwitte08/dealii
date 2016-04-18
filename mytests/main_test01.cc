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

#include <iostream>
#include <vector>

// RHS INTEGRATOR
template <int dim>
class RHSIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
 public:
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
            dealii::MeshWorker::DoFInfo<dim> &dinfo2,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
};

// MAIN
int main ()
{
  const unsigned int dim = 2;
  const unsigned int degree = 1;

  dealii::Triangulation<dim> tria;
  const dealii::FE_DGQ<dim> fe{degree};
  dealii::DoFHandler<dim> dof_handler{tria};
  const dealii::MappingQ1<dim> mapping;
  dealii::Vector<double> rhs;

  dealii::GridGenerator::hyper_cube (tria,0.,1.);
  tria.refine_global(10);

  dof_handler.distribute_dofs(fe);
  //dof_handler.initialize_local_block_info();

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

  RHSIntegrator<dim> rhs_integrator;

  dealii::MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(),
						 dof_handler.end(),
						 dof_info, info_box,
						 rhs_integrator, rhs_assembler);
  //rhs.print(std::cout);
  return 0;
}

// IMPLEMENTATION: RHS_INTEGRATOR
template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  std::vector<double> f;
  f.resize(info.fe_values(0).n_quadrature_points,1.);
  dealii::LocalIntegrators::L2::L2(dinfo.vector(0).block(0),
				   info.fe_values(0),
				   f);
}

template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &, typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template <int dim>
void RHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
                              dealii::MeshWorker::DoFInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}
