// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2020 by the deal.II authors
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


#include <deal.II/base/polynomials_raviart_thomas_new.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/thread_management.h>

#include <iomanip>
#include <iostream>
#include <memory>

// TODO[WB]: This class is not thread-safe: it uses mutable member variables
// that contain temporary state. this is not what one would want when one uses a
// finite element object in a number of different contexts on different threads:
// finite element objects should be stateless
// TODO:[GK] This can be achieved by writing a function in Polynomial space
// which does the rotated fill performed below and writes the data into the
// right data structures. The same function would be used by Nedelec
// polynomials.

DEAL_II_NAMESPACE_OPEN


template <int dim>
PolynomialsRaviartThomas_new<dim>::PolynomialsRaviartThomas_new(
  const unsigned int k)
  : TensorPolynomialsBase<dim>(k, n_polynomials(k))
  , polynomial_space(std::move([&]() {
    std::vector<AnisotropicPolynomials<dim>> polynomial_space_in;
    for (unsigned int comp = 0; comp < dim; ++comp)
      {
        std::vector<std::vector<Polynomials::Polynomial<double>>>
          scalar_polynomials;
        for (unsigned int d = 0; d < dim; ++d)
          scalar_polynomials.emplace_back(
            comp == d ? make_univariate_polynomials_high(k) :
                        make_univariate_polynomials_low(k));
        polynomial_space_in.emplace_back(scalar_polynomials);
      }
    return polynomial_space_in;
  }()))
{
  AssertDimension(polynomial_space.size(), dim);
}


template <int dim>
void
PolynomialsRaviartThomas_new<dim>::evaluate(
  const Point<dim> &           unit_point,
  std::vector<Tensor<1, dim>> &values,
  std::vector<Tensor<2, dim>> &grads,
  std::vector<Tensor<3, dim>> &grad_grads,
  std::vector<Tensor<4, dim>> &third_derivatives,
  std::vector<Tensor<5, dim>> &fourth_derivatives) const
{
  Assert(values.size() == this->n() || values.size() == 0,
         ExcDimensionMismatch(values.size(), this->n()));
  Assert(grads.size() == this->n() || grads.size() == 0,
         ExcDimensionMismatch(grads.size(), this->n()));
  Assert(grad_grads.size() == this->n() || grad_grads.size() == 0,
         ExcDimensionMismatch(grad_grads.size(), this->n()));
  Assert(third_derivatives.size() == this->n() || third_derivatives.size() == 0,
         ExcDimensionMismatch(third_derivatives.size(), this->n()));
  Assert(fourth_derivatives.size() == this->n() ||
           fourth_derivatives.size() == 0,
         ExcDimensionMismatch(fourth_derivatives.size(), this->n()));

  // have a few scratch
  // arrays. because we don't want to
  // re-allocate them every time this
  // function is called, we make them
  // static. however, in return we
  // have to ensure that the calls to
  // the use of these variables is
  // locked with a mutex. if the
  // mutex is removed, several tests
  // (notably
  // deal.II/create_mass_matrix_05)
  // will start to produce random
  // results in multithread mode
  static std::mutex           mutex;
  std::lock_guard<std::mutex> lock(mutex);

  /// TODO maybe use non-static data members which are resized during
  /// construction?
  static std::vector<double>         p_values;
  static std::vector<Tensor<1, dim>> p_grads;
  static std::vector<Tensor<2, dim>> p_grad_grads;
  static std::vector<Tensor<3, dim>> p_third_derivatives;
  static std::vector<Tensor<4, dim>> p_fourth_derivatives;

  /// assuming isotropy among vector components
  const unsigned int n_sub = polynomial_space[0].n();
  AssertDimension(this->n(), dim * n_sub);

  p_values.resize((values.size() == 0) ? 0 : n_sub);
  p_grads.resize((grads.size() == 0) ? 0 : n_sub);
  p_grad_grads.resize((grad_grads.size() == 0) ? 0 : n_sub);
  p_third_derivatives.resize((third_derivatives.size() == 0) ? 0 : n_sub);
  p_fourth_derivatives.resize((fourth_derivatives.size() == 0) ? 0 : n_sub);

  for (unsigned int d = 0; d < dim; ++d)
    {
      const unsigned int offset_comp     = d * n_sub;
      const auto &       this_poly_space = polynomial_space[d];

      this_poly_space.evaluate(unit_point,
                               p_values,
                               p_grads,
                               p_grad_grads,
                               p_third_derivatives,
                               p_fourth_derivatives);

      auto dst_value = values.begin() + offset_comp;
      for (auto src_value = p_values.cbegin(); src_value != p_values.cend();
           ++dst_value, ++src_value)
        (*dst_value)[d] = (*src_value);
      auto dst_grad = grads.begin() + offset_comp;
      for (auto src_grad = p_grads.cbegin(); src_grad != p_grads.cend();
           ++dst_grad, ++src_grad)
        (*dst_grad)[d] = (*src_grad);
      auto dst_grad_grad = grad_grads.begin() + offset_comp;
      for (auto src_grad_grad = p_grad_grads.cbegin();
           src_grad_grad != p_grad_grads.cend();
           ++dst_grad_grad, ++src_grad_grad)
        (*dst_grad_grad)[d] = (*src_grad_grad);
      auto dst_third_derivative = third_derivatives.begin() + offset_comp;
      for (auto src_third_derivative = p_third_derivatives.cbegin();
           src_third_derivative != p_third_derivatives.cend();
           ++dst_third_derivative, ++src_third_derivative)
        (*dst_third_derivative)[d] = (*src_third_derivative);
      auto dst_fourth_derivative = fourth_derivatives.begin() + offset_comp;
      for (auto src_fourth_derivative = p_fourth_derivatives.cbegin();
           src_fourth_derivative != p_fourth_derivatives.cend();
           ++dst_fourth_derivative, ++src_fourth_derivative)
        (*dst_fourth_derivative)[d] = (*src_fourth_derivative);
    }
}


template <int dim>
unsigned int
PolynomialsRaviartThomas_new<dim>::n_polynomials(const unsigned int k)
{
  if (dim == 1)
    return k + 1;
  if (dim == 2)
    return 2 * (k + 1) * (k + 2);
  if (dim == 3)
    return 3 * (k + 1) * (k + 1) * (k + 2);

  Assert(false, ExcNotImplemented());
  return 0;
}


template <int dim>
std::unique_ptr<TensorPolynomialsBase<dim>>
PolynomialsRaviartThomas_new<dim>::clone() const
{
  return std::make_unique<PolynomialsRaviartThomas_new<dim>>(*this);
}



template <int dim>
std::vector<Polynomials::Polynomial<double>>
PolynomialsRaviartThomas_new<dim>::make_univariate_polynomials_low(
  const unsigned int k)
{
  return k == 0 ? Polynomials::Legendre::generate_complete_basis(0) :
                  Polynomials::LagrangeEquidistant::generate_complete_basis(k);
}


template <int dim>
std::vector<Polynomials::Polynomial<double>>
PolynomialsRaviartThomas_new<dim>::make_univariate_polynomials_high(
  const unsigned int k)
{
  return make_univariate_polynomials_low(k + 1);
}


template class PolynomialsRaviartThomas_new<1>;
template class PolynomialsRaviartThomas_new<2>;
template class PolynomialsRaviartThomas_new<3>;


DEAL_II_NAMESPACE_CLOSE
