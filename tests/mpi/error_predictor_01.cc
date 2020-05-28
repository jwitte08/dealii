// ---------------------------------------------------------------------
//
// Copyright (C) 2019 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------



// Test to check if ErrorPredictor works in parallel with hp::DoFHandler.
// This tests is based on hp/error_prediction.cc


#include <deal.II/distributed/error_predictor.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/dof_handler.h>

#include <deal.II/lac/vector.h>

#include "../tests.h"


template <int dim>
void
test()
{
  const unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int n_cells = 4;

  // ------ setup ------
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::vector<unsigned int>                 rep(dim, 1);
  rep[0] = n_cells;
  Point<dim> p1, p2;
  for (unsigned int d = 0; d < dim; ++d)
    {
      p1[d] = 0;
      p2[d] = (d == 0) ? n_cells : 1;
    }
  GridGenerator::subdivided_hyper_rectangle(tria, rep, p1, p2);

  {
    auto first = tria.begin(0);
    if (first->id().to_string() == "0_0:")
      first->set_refine_flag();
  }
  tria.execute_coarsening_and_refinement();

  hp::FECollection<dim> fes;
  for (unsigned int d = 1; d <= 3; ++d)
    fes.push_back(FE_Q<dim>(d));

  hp::DoFHandler<dim> dh(tria);
  dh.set_fe(fes);
  for (auto cell = dh.begin(0); cell != dh.end(0); ++cell)
    if (cell->id().to_string() == "0_0:")
      {
        // h-coarsening
        for (unsigned int i = 0; i < cell->n_children(); ++i)
          if (cell->child(i)->is_locally_owned())
            cell->child(i)->set_coarsen_flag();
      }
    else if (cell->id().to_string() == "1_0:")
      {
        // h-refinement
        if (cell->is_locally_owned())
          cell->set_refine_flag();
      }
    else if (cell->id().to_string() == "2_0:")
      {
        // p-refinement
        if (cell->is_locally_owned())
          cell->set_future_fe_index(2);
      }

  // ----- prepare error indicators -----
  Vector<float> error_indicators(tria.n_active_cells());
  for (unsigned int i = 0; i < error_indicators.size(); ++i)
    error_indicators(i) = 10.;

  // ----- verify ------
  deallog << "pre_adaptation" << std::endl;
  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        deallog << " cell:" << cell->id().to_string()
                << " fe_deg:" << cell->get_fe().degree
                << " error:" << error_indicators[cell->active_cell_index()];

        if (cell->coarsen_flag_set())
          deallog << " coarsening";
        else if (cell->refine_flag_set())
          deallog << " refining";

        if (cell->future_fe_index_set())
          deallog << " future_fe_deg:" << fes[cell->future_fe_index()].degree;

        deallog << std::endl;
      }

  // ----- execute adaptation -----
  parallel::distributed::ErrorPredictor<dim> predictor(dh);

  predictor.prepare_for_coarsening_and_refinement(error_indicators,
                                                  /*gamma_p=*/0.5,
                                                  /*gamma_h=*/1.,
                                                  /*gamma_n=*/1.);
  tria.execute_coarsening_and_refinement();

  Vector<float> predicted_errors(tria.n_active_cells());
  predictor.unpack(predicted_errors);

  // ------ verify ------
  deallog << "post_adaptation" << std::endl;
  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      deallog << " cell:" << cell->id().to_string()
              << " predicted:" << predicted_errors(cell->active_cell_index())
              << std::endl;

  // make sure no processor is hanging
  MPI_Barrier(MPI_COMM_WORLD);

  deallog << "OK" << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    log;

  deallog.push("2d");
  test<2>();
  deallog.pop();
  deallog.push("3d");
  test<3>();
  deallog.pop();
}
