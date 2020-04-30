// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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



// Check the correctness of the 1d evaluation for Raviart-Thomas elements, using
// FEEvaluationImpl with element type 'raviart_thomas': We check the
// interpolation of values on the reference element.

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <sstream>

#include "../tests.h"


/*
 * The polynomial
 *
 *    (p(x,y,z)+x, p(x,y,z)+y, p(x,y,z)+z)^T with p in [Q_k]^dim
 *
 * is exactly interpolated by RT_k elements.
 */
template <int dim>
class RTPoly : public Function<dim, double>
{
public:
  RTPoly(const unsigned int degree)
    : Function<dim, double>(dim)
    , Q(Polynomials::Legendre::generate_complete_basis(degree))
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    return Q.compute_value(Q.n() - 1, p) + p[component];
  }

  TensorProductPolynomials<dim> Q;
};



/*
 * Note that the parameter fe_degree here is by one higher than the actual RT
 * degree (in the deal.II common notation).
 */
template <int dim,
          int fe_degree,
          typename Number           = double,
          typename VectorizedNumber = VectorizedArray<Number>>
void
compare_interpolation(const Function<dim> &f, const std::string func_dscr)
{
  Triangulation<dim> triangulation;
  const auto         rt_degree = fe_degree - 1; // see documentation of FE
  FE_RaviartThomasNodal_new<dim> fe(rt_degree);
  DoFHandler<dim>                dof_handler(triangulation);

  GridGenerator::hyper_cube(triangulation, 0., 1.);
  dof_handler.distribute_dofs(fe);

  const unsigned int     n_dofs_per_cell = fe.dofs_per_cell;
  constexpr unsigned int n_q_points_1d   = fe_degree + 1;
  deallog << fe.get_name() << " with " << n_dofs_per_cell << " dofs"
          << std::endl;
  QGauss<dim> quadrature(n_q_points_1d);
  deallog << "Numerical integration with " << quadrature.size()
          << " quadrature points" << std::endl;

  Vector<Number> dof_values(n_dofs_per_cell);
  VectorTools::interpolate(dof_handler, f, dof_values);

  /// using FEValues as reference
  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  AssertDimension(n_dofs_per_cell, dof_values.size());
  std::vector<Tensor<1, dim, Number>> quad_rvalues(quadrature.size());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      for (auto q = 0U; q < quadrature.size(); ++q)
        {
          Tensor<1, dim, Number> val;
          for (auto i = 0U; i < n_dofs_per_cell; ++i)
            for (auto d = 0; d < dim; ++d)
              val[d] +=
                dof_values[i] * fe_values.shape_value_component(i, q, d);
          quad_rvalues[q] = val;
        }
    }

  /// actual values of function f
  std::vector<Tensor<1, dim, Number>> quad_avalues(quadrature.size());
  for (auto q = 0U; q < quadrature.size(); ++q)
    {
      const auto &q_points = quadrature.get_points();
      for (auto d = 0U; d < dim; ++d)
        quad_avalues[q][d] = f.value(q_points[q], d);
    }

  /// matrix-free interpolation via FEEvaluationImpl
  using namespace internal;
  MatrixFreeFunctions::ShapeInfo<Number> shape_info;
  QGauss<1>                              quadrature_1d(n_q_points_1d);
  shape_info.reinit(quadrature_1d, fe, 0);
  FEEvaluationImpl<MatrixFreeFunctions::ElementType::raviart_thomas,
                   dim,
                   fe_degree,
                   n_q_points_1d,
                   dim,
                   Number>
                        feev_impl;
  AlignedVector<Number> dof_values_lxco(dof_values.size());
  // deallog << "dealii vs. lexicographic numbering" << std::endl;
  for (auto i = 0U; i < dof_values.size(); ++i)
    {
      const auto ii = shape_info.lexicographic_numbering[i];
      // deallog << i << " vs. " << ii << std::endl;
      dof_values_lxco[ii] = dof_values[i];
    }
  AlignedVector<Number> quad_values_flat(quadrature.size() * dim);
  AlignedVector<Number> scratchpad(quadrature.size() * dim);
  feev_impl.evaluate(shape_info,
                     dof_values_lxco.data(),
                     quad_values_flat.data(),
                     nullptr,
                     nullptr,
                     scratchpad.data(),
                     true,
                     false,
                     false);
  std::vector<Tensor<1, dim, Number>> quad_values(quadrature.size());
  for (auto q = 0U; q < quadrature.size(); ++q)
    for (auto c = 0U; c < dim; ++c)
      quad_values[q][c] = quad_values_flat[quadrature.size() * c + q];

  deallog << "Compare matrix-free interpolation (left) of " << func_dscr
          << " (right):" << std::endl;
  const auto numeric_eps =
    std::pow(10, std::log10(std::numeric_limits<Number>::epsilon()) / 2);
  for (auto q = 0U; q < quadrature.size(); ++q)
    {
      deallog << quad_values[q] << " vs. " << quad_rvalues[q] << " vs. "
              << quad_avalues[q] << std::endl;
      Assert((quad_values[q] - quad_avalues[q]).norm() < numeric_eps,
             ExcMessage("Mismatching values."));
    }
}



int
main()
{
  initlog();

  {
    // 2D
    Functions::ConstantFunction<2, double> f(2., 2);
    compare_interpolation<2, 1>(f, "a constant function");
    compare_interpolation<2, 2>(f, "a constant function");
    compare_interpolation<2, 3>(f, "a constant function");

    // 3D
    Functions::ConstantFunction<3, double> ff(2., 3);
    compare_interpolation<3, 1>(ff, "a constant function");
    compare_interpolation<3, 2>(ff, "a constant function");
    compare_interpolation<3, 3>(ff, "a constant function");
  }

  {
    { // 2D
      constexpr int dim = 2;
      RTPoly<dim>   rt0(0);
      compare_interpolation<dim, 1>(rt0, "a polynomial in RT space");
      RTPoly<dim> rt1(1);
      compare_interpolation<dim, 2>(rt1, "a polynomial in RT space");
      RTPoly<dim> rt2(2);
      compare_interpolation<dim, 3>(rt2, "a polynomial in RT space");
    }

    { // 3D
      constexpr int dim = 3;
      RTPoly<dim>   rt0(0);
      compare_interpolation<dim, 1>(rt0, "a polynomial in RT space");
      RTPoly<dim> rt1(1);
      compare_interpolation<dim, 2>(rt1, "a polynomial in RT space");
      RTPoly<dim> rt2(2);
      compare_interpolation<dim, 3>(rt2, "a polynomial in RT space");
    }
  }

  for (auto direction = 0U; direction < 2; ++direction)
    {
      Tensor<1, 2> exp;
      for (auto d = 0U; d < 2; ++d)
        exp[d] = d == direction ? 1 : 0;
      Functions::Monomial<2> f(exp, 2);
      const std::string      xyz[3] = {"x", "y", "z"};
      compare_interpolation<2, 2>(f, "a function linear in " + xyz[direction]);
    }

  return 0;
}
