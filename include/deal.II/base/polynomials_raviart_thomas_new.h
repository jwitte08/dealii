// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2019 by the deal.II authors
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

#ifndef dealii_polynomials_raviart_thomas_new_h
#define dealii_polynomials_raviart_thomas_new_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_polynomials_base.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * This class implements the <i>H<sup>div</sup></i>-conforming, vector-valued
 * Raviart-Thomas polynomials as described in the book by Brezzi and Fortin.
 *
 * The Raviart-Thomas polynomials are constructed such that the divergence is
 * in the tensor product polynomial space <i>Q<sub>k</sub></i>. Therefore, the
 * polynomial order of each component must be one order higher in the
 * corresponding direction, yielding the polynomial spaces
 * <i>(Q<sub>k+1,k</sub>, Q<sub>k,k+1</sub>)</i> and <i>(Q<sub>k+1,k,k</sub>,
 * Q<sub>k,k+1,k</sub>, Q<sub>k,k,k+1</sub>)</i> in 2D and 3D, resp.
 *
 * @ingroup Polynomials
 */
template <int dim>
class PolynomialsRaviartThomas_new : public TensorPolynomialsBase<dim>
{
public:
  /**
   * Constructor. Creates all basis functions for Raviart-Thomas polynomials
   * of given degree.
   *
   * @arg k: the degree of the Raviart-Thomas-space, which is the degree of
   * the largest tensor product polynomial space <i>Q<sub>k</sub></i>
   * contains.
   */
  PolynomialsRaviartThomas_new(const unsigned int k);

  /**
   * Compute the value and the first and second derivatives of each Raviart-
   * Thomas polynomial at @p unit_point.
   *
   * The size of the vectors must either be zero or equal <tt>n()</tt>.  In
   * the first case, the function will not compute these values.
   *
   * If you need values or derivatives of all tensor product polynomials then
   * use this function, rather than using any of the <tt>compute_value</tt>,
   * <tt>compute_grad</tt> or <tt>compute_grad_grad</tt> functions, see below,
   * in a loop over all tensor product polynomials.
   */
  void
  evaluate(const Point<dim> &           unit_point,
           std::vector<Tensor<1, dim>> &values,
           std::vector<Tensor<2, dim>> &grads,
           std::vector<Tensor<3, dim>> &grad_grads,
           std::vector<Tensor<4, dim>> &third_derivatives,
           std::vector<Tensor<5, dim>> &fourth_derivatives) const override;

  /**
   * Return the name of the space, which is <tt>RaviartThomas</tt>.
   */
  std::string
  name() const override;

  /**
   * Return the number of polynomials in the space <tt>RT(degree)</tt> without
   * requiring to build an object of PolynomialsRaviartThomas_new. This is
   * required by the FiniteElement classes.
   */
  static unsigned int
  n_polynomials(const unsigned int degree);

  /**
   * @copydoc TensorPolynomialsBase::clone()
   */
  virtual std::unique_ptr<TensorPolynomialsBase<dim>>
  clone() const override;

  /**
   * Generating a basis of univariate polynomials of lower polynomial degree k
   * the anisotropic polynomial space is based on.
   */
  static std::vector<Polynomials::Polynomial<double>>
  make_univariate_polynomials_low(unsigned int k);

  /**
   * Generating a basis of univariate polynomials of higher polynomial degree
   * (k+1) the anisotropic polynomial space is based on.
   */
  static std::vector<Polynomials::Polynomial<double>>
  make_univariate_polynomials_high(unsigned int k);

private:
  /**
   * An object representing the polynomial space for a single component. We
   * can re-use it by rotating the coordinates of the evaluation point.
   */
  std::vector<AnisotropicPolynomials<dim>> polynomial_space;
};


template <int dim>
inline std::string
PolynomialsRaviartThomas_new<dim>::name() const
{
  return "RaviartThomas";
}


DEAL_II_NAMESPACE_CLOSE

#endif
