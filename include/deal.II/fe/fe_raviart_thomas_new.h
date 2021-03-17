// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2019 by the deal.II authors
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

#ifndef dealii_fe_raviart_thomas_new_h
#define dealii_fe_raviart_thomas_new_h

#include <deal.II/base/config.h>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/base/polynomials_raviart_thomas_new.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_poly_tensor.h>

#include <deal.II/matrix_free/shape_info.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

/*!@addtogroup fe */
/*@{*/

/**
 * Implementation of Raviart-Thomas (RT) elements. The Raviart-Thomas space
 * is designed to solve problems in which the solution only lives in the
 * space
 * $H^\text{div}=\{ {\mathbf u} \in L_2: \text{div}\, {\mathbf u} \in L_2\}$,
 * rather than in the more commonly used space
 * $H^1=\{ u \in L_2: \nabla u \in L_2\}$. In other words, the solution must
 * be a vector field whose divergence is square integrable, but for which the
 * gradient may not be square integrable. The typical application for this
 * space (and these elements) is to the mixed formulation of the Laplace
 * equation and related situations, see for example step-20. The defining
 * characteristic of functions in $H^\text{div}$ is that they are in
 * general discontinuous -- but that if you draw a line in 2d (or a
 * surface in 3d), then the <i>normal</i> component of the vector
 * field must be continuous across the line (or surface) even though
 * the tangential component may not be. As a consequence, the
 * Raviart-Thomas element is constructed in such a way that (i) it is
 * @ref vector_valued "vector-valued", (ii) the shape functions are
 * discontinuous, but (iii) the normal component of the vector field
 * represented by each shape function is continuous across the faces
 * of cells.
 *
 * Other properties of the Raviart-Thomas element are that (i) it is
 * @ref GlossPrimitive "not a primitive element"; (ii) the shape functions
 * are defined so that certain integrals over the faces are either zero
 * or one, rather than the common case of certain point values being
 * either zero or one. (There is, however, the FE_RaviartThomasNodal
 * element that uses point values.)
 *
 * We follow the commonly used -- though confusing -- definition of the "degree"
 * of RT elements. Specifically, the "degree" of the element denotes
 * the polynomial degree of the <i>largest complete polynomial subspace</i>
 * contained in the finite element space, even if the space may contain shape
 * functions of higher polynomial degree. The lowest order element is
 * consequently FE_RaviartThomas_new(0), i.e., the Raviart-Thomas element "of
 * degree zero", even though the functions of this space are in general
 * polynomials of degree one in each variable. This choice of "degree"
 * implies that the approximation order of the function itself is
 * <i>degree+1</i>, as with usual polynomial spaces. The numbering so chosen
 * implies the sequence
 * @f[
 *   Q_{k+1}
 *   \stackrel{\text{grad}}{\rightarrow}
 *   \text{Nedelec}_k
 *   \stackrel{\text{curl}}{\rightarrow}
 *   \text{RaviartThomas}_k
 *   \stackrel{\text{div}}{\rightarrow}
 *   DGQ_{k}
 * @f]
 *
 * This class is not implemented for the codimension one case (<tt>spacedim !=
 * dim</tt>).
 *
 *
 * <h3>Interpolation</h3>
 *
 * The
 * @ref GlossInterpolation "interpolation"
 * operators associated with the RT element are constructed such that
 * interpolation and computing the divergence are commuting operations. We
 * require this from interpolating arbitrary functions as well as the
 * #restriction matrices.  It can be achieved by two interpolation schemes,
 * the simplified one in FE_RaviartThomasNodal and the original one here:
 *
 * <h4>Node values on edges/faces</h4>
 *
 * On edges or faces, the
 * @ref GlossNodes "node values"
 * are the moments of the normal component of the interpolated function with
 * respect to the traces of the RT polynomials. Since the normal trace of the
 * RT space of degree <i>k</i> on an edge/face is the space
 * <i>Q<sub>k</sub></i>, the moments are taken with respect to this space.
 *
 * <h4>Interior node values</h4>
 *
 * Higher order RT spaces have interior nodes. These are moments taken with
 * respect to the gradient of functions in <i>Q<sub>k</sub></i> on the cell
 * (this space is the matching space for RT<sub>k</sub> in a mixed
 * formulation).
 *
 * <h4>Generalized support points</h4>
 *
 * The node values above rely on integrals, which will be computed by
 * quadrature rules themselves. The generalized support points are a set of
 * points such that this quadrature can be performed with sufficient accuracy.
 * The points needed are those of QGauss<sub>k+1</sub> on each face as well as
 * QGauss<sub>k+1</sub> in the interior of the cell (or none for
 * RT<sub>0</sub>).
 */
template <int dim>
class FE_RaviartThomas_new : public FE_PolyTensor<dim>
{
public:
  /**
   * Constructor for the Raviart-Thomas element of degree @p p.
   */
  FE_RaviartThomas_new(const unsigned int p);

  /// NEW create h2l transformation
  std::vector<unsigned int>
  make_hierarchical_to_lexicographic_index_map(
    const unsigned int fe_degree) const;

  /// NEW the data filled suffices to reflect the tensor structure of this
  /// finite element
  template <typename Number>
  void
  fill_shape_info(internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                  Quadrature<1>                                     quad) const;

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_RaviartThomas_new<dim>(degree)</tt>, with @p dim and @p degree
   * replaced by appropriate values.
   */
  virtual std::string
  get_name() const override;

  // documentation inherited from the base class
  virtual std::unique_ptr<FiniteElement<dim, dim>>
  clone() const override;

  /**
   * This function returns @p true, if the shape function @p shape_index has
   * non-zero function values somewhere on the face @p face_index.
   *
   * Right now, this is only implemented for RT0 in 1D. Otherwise, returns
   * always @p true.
   */
  virtual bool
  has_support_on_face(const unsigned int shape_index,
                      const unsigned int face_index) const override;

  // // documentation inherited from the base class
  // virtual void
  // convert_generalized_support_point_values_to_dof_values(
  //   const std::vector<Vector<double>> &support_point_values,
  //   std::vector<double> &              nodal_values) const override;

  /**
   * Return a list of constant modes of the element. This method is currently
   * not correctly implemented because it returns ones for all components.
   */
  virtual std::pair<Table<2, bool>, std::vector<unsigned int>>
  get_constant_modes() const override;

  virtual std::size_t
  memory_consumption() const override;

private:
  /**
   * Only for internal use. Its full name is @p get_dofs_per_object_vector
   * function and it creates the @p dofs_per_object vector that is needed
   * within the constructor to be passed to the constructor of @p
   * FiniteElementData.
   */
  static std::vector<unsigned int>
  get_dpo_vector(const unsigned int degree);

  /**
   * Initialize the @p generalized_support_points field of the FiniteElement
   * class and fill the tables with interpolation weights (#boundary_weights
   * and #interior_weights). Called from the constructor.
   */
  void
  initialize_support_points(const unsigned int rt_degree);

  /**
   * Initialize the interpolation from functions on refined mesh cells onto
   * the father cell. According to the philosophy of the Raviart-Thomas
   * element, this restriction operator preserves the divergence of a function
   * weakly.
   */
  void
  initialize_restriction();

  double
  evaluate_node_functional_kplus1(const unsigned int                     i,
                                  const Polynomials::Polynomial<double> &poly,
                                  const Quadrature<1> &quad) const;

  template <bool is_nodal>
  double
  evaluate_node_functional_kplus1_impl(
    const unsigned int             i,
    const ArrayView<const double> &poly_values,
    const Quadrature<1> &          quad) const;

  double
  evaluate_node_functional_k(const unsigned int                     i,
                             const Polynomials::Polynomial<double> &poly,
                             const Quadrature<1> &                  quad) const;

  double
  evaluate_node_functional_k_impl(const unsigned int             i,
                                  const ArrayView<const double> &poly_values,
                                  const Quadrature<1> &          quad) const;

  /**
   * These are the factors multiplied to a function in the
   * #generalized_face_support_points when computing the integration. They are
   * organized such that there is one row for each generalized face support
   * point and one column for each degree of freedom on the face.
   *
   * See the
   * @ref GlossGeneralizedSupport "glossary entry on generalized support points"
   * for more information.
   */
  Table<2, double> boundary_weights;

  /**
   * Precomputed factors for interpolation of interior degrees of freedom. The
   * rationale for this Table is the same as for #boundary_weights. Only, this
   * table has a third coordinate for the space direction of the component
   * evaluated.
   */
  Table<3, double> interior_weights;

  /// NEW hierarchical-to-lexicographic index mapping
  std::vector<unsigned int> h2l;

  /// NEW lexicographic-to-hierarchical index mapping
  std::vector<unsigned int> l2h;

  /// NEW univariate raw polynomial basis of degree up to (k+1)
  const std::vector<Polynomials::Polynomial<double>> raw_polynomials_kplus1;

  /// NEW univariate raw polynomial basis of degree up to k
  const std::vector<Polynomials::Polynomial<double>> raw_polynomials_k;

  /// NEW univariate polynomial basis of degree up to k generating moment-based
  /// node functionals
  const std::vector<Polynomials::Polynomial<double>> node_polynomials_k;

  /// NEW univariate polynomial basis of degree up to (k-1) generating
  /// moment-based node functionals
  const std::vector<Polynomials::Polynomial<double>> node_polynomials_kminus1;

  /// NEW inverse node values which determine the transpose of the
  /// transformation matrix from raw polynomials to the univariate shape
  /// function basis of degree up to (k+1). The first and last node functional
  /// is nodal in 0 and 1, resp.. All remaining functionals are moments on the
  /// unit interval.
  FullMatrix<double> inverse_node_value_matrix_kplus1;

  /// NEW inverse node values which determine the transpose the transformation
  /// matrix from raw polynomials to the univariate shape function basis of
  /// degree up to k. All node functionals are moments on the unit interval.
  FullMatrix<double> inverse_node_value_matrix_k;

  // Allow access from other dimensions.
  template <int dim1>
  friend class FE_RaviartThomas_new;
};



/**
 * The Raviart-Thomas elements with node functionals defined as point values
 * in Gauss points.
 *
 * <h3>Description of node values</h3>
 *
 * For this Raviart-Thomas element, the node values are not cell and face
 * moments with respect to certain polynomials, but the values in quadrature
 * points. Following the general scheme for numbering degrees of freedom, the
 * node values on edges are first, edge by edge, according to the natural
 * ordering of the edges of a cell. The interior degrees of freedom are last.
 *
 * For an RT-element of degree <i>k</i>, we choose <i>(k+1)<sup>d-1</sup></i>
 * Gauss points on each face. These points are ordered lexicographically with
 * respect to the orientation of the face. This way, the normal component
 * which is in <i>Q<sub>k</sub></i> is uniquely determined. Furthermore, since
 * this Gauss-formula is exact on <i>Q<sub>2k+1</sub></i>, these node values
 * correspond to the exact integration of the moments of the RT-space.
 *
 * In the interior of the cells, the moments are with respect to an
 * anisotropic <i>Q<sub>k</sub></i> space, where the test functions are one
 * degree lower in the direction corresponding to the vector component under
 * consideration. This is emulated by using an anisotropic Gauss formula for
 * integration.
 *
 * @todo The current implementation is for Cartesian meshes only. You must use
 * MappingCartesian.
 *
 * @todo Even if this element is implemented for two and three space
 * dimensions, the definition of the node values relies on consistently
 * oriented faces in 3D. Therefore, care should be taken on complicated
 * meshes.
 *
 * @note The degree stored in the member variable
 * FiniteElementData<dim>::degree is higher by one than the constructor
 * argument!
 */
template <int dim>
class FE_RaviartThomasNodal_new : public FE_PolyTensor<dim>
{
public:
  /**
   * Constructor for the Raviart-Thomas element of degree @p p.
   */
  FE_RaviartThomasNodal_new(const unsigned int p);

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_RaviartThomasNodal_new<dim>(degree)</tt>, with @p dim and @p
   * degree replaced by appropriate values.
   */
  virtual std::string
  get_name() const override;

  // documentation inherited from the base class
  virtual std::unique_ptr<FiniteElement<dim, dim>>
  clone() const override;

  virtual void
  convert_generalized_support_point_values_to_dof_values(
    const std::vector<Vector<double>> &support_point_values,
    std::vector<double> &              nodal_values) const override;

  virtual void
  get_face_interpolation_matrix(const FiniteElement<dim> &source,
                                FullMatrix<double> &      matrix,
                                const unsigned int face_no = 0) const override;

  virtual void
  get_subface_interpolation_matrix(
    const FiniteElement<dim> &source,
    const unsigned int        subface,
    FullMatrix<double> &      matrix,
    const unsigned int        face_no = 0) const override;
  virtual bool
  hp_constraints_are_implemented() const override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_vertex_dof_identities(const FiniteElement<dim> &fe_other) const override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_line_dof_identities(const FiniteElement<dim> &fe_other) const override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_quad_dof_identities(const FiniteElement<dim> &fe_other,
                         const unsigned int        face_no = 0) const override;

  /**
   * @copydoc FiniteElement::compare_for_domination()
   */
  virtual FiniteElementDomination::Domination
  compare_for_domination(const FiniteElement<dim> &fe_other,
                         const unsigned int codim = 0) const override final;

  double
  shape_value_component_as_tensor_product(const unsigned int i,
                                          const Point<dim> & p,
                                          const unsigned int component);
  template <typename Number>
  void
  fill_shape_info(internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                  Quadrature<1>                                     quad) const
  {
    using namespace internal::MatrixFreeFunctions;

    /// to be on the safe side set shape_info to its default
    shape_info = internal::MatrixFreeFunctions::ShapeInfo<Number>{};

    AssertDimension(shape_info.data.size(), 0U); // double-check
    shape_info.data.reserve(2U);
    /// c++17: auto & higher = shape_info.data.emplace_back();
    shape_info.data.emplace_back();
    shape_info.data.emplace_back();
    UnivariateShapeData<Number> &higher = shape_info.data.front();
    UnivariateShapeData<Number> &lower  = shape_info.data.back();

    lower.element_type              = tensor_symmetric;
    higher.element_type             = tensor_symmetric;
    lower.fe_degree                 = nodal_basis_of_low.size() - 1;
    higher.fe_degree                = nodal_basis_of_high.size() - 1;
    lower.quadrature                = quad;
    higher.quadrature               = quad;
    lower.n_q_points_1d             = quad.size();
    higher.n_q_points_1d            = quad.size();
    lower.nodal_at_cell_boundaries  = false;
    higher.nodal_at_cell_boundaries = true;

    lower.shape_values =
      AlignedVector<Number>((lower.fe_degree + 1) * lower.n_q_points_1d); //+1
    higher.shape_values =
      AlignedVector<Number>((higher.fe_degree + 1) * higher.n_q_points_1d);
    lower.shape_gradients =
      AlignedVector<Number>((lower.fe_degree + 1) * lower.n_q_points_1d);
    higher.shape_gradients =
      AlignedVector<Number>((higher.fe_degree + 1) * higher.n_q_points_1d);
    lower.shape_hessians =
      AlignedVector<Number>((lower.fe_degree + 1) * lower.n_q_points_1d);
    higher.shape_hessians =
      AlignedVector<Number>((higher.fe_degree + 1) * higher.n_q_points_1d);
    lower.shape_data_on_face[0].resize(3 * (lower.fe_degree + 1));
    lower.shape_data_on_face[1].resize(3 * (lower.fe_degree + 1));
    higher.shape_data_on_face[0].resize(3 * (higher.fe_degree + 1));
    higher.shape_data_on_face[1].resize(3 * (higher.fe_degree + 1));

    const auto          n_q_points_1d = quad.size();
    std::vector<double> q_points;
    std::transform(quad.get_points().cbegin(),
                   quad.get_points().cend(),
                   std::back_inserter(q_points),
                   [](const auto &point) { return point(0); });
    for (std::size_t i = 0; i < (lower.fe_degree + 1); ++i)
      {
        for (std::size_t j = 0; j < n_q_points_1d; ++j)
          {
            lower.shape_values[i * n_q_points_1d + j] =
              nodal_basis_of_low[i].value(q_points[j]);
            lower.shape_gradients[i * n_q_points_1d + j] =
              nodal_basis_of_low[i].derivative().value(q_points[j]);
            lower.shape_hessians[i * n_q_points_1d + j] =
              nodal_basis_of_low[i].derivative().derivative().value(
                q_points[j]);
          }
        lower.shape_data_on_face[0][i] = nodal_basis_of_low[i].value(0);
        lower.shape_data_on_face[0][i + (lower.fe_degree + 1)] =
          nodal_basis_of_low[i].derivative().value(0);
        lower.shape_data_on_face[0][i + 2 * (lower.fe_degree + 1)] =
          nodal_basis_of_low[i].derivative().derivative().value(0);
        lower.shape_data_on_face[1][i] = nodal_basis_of_low[i].value(1);
        lower.shape_data_on_face[1][i + (lower.fe_degree + 1)] =
          nodal_basis_of_low[i].derivative().value(1);
        lower.shape_data_on_face[1][i + 2 * (lower.fe_degree + 1)] =
          nodal_basis_of_low[i].derivative().derivative().value(1);
      }
    for (std::size_t i = 0; i < (higher.fe_degree + 1); ++i)
      {
        for (std::size_t j = 0; j < n_q_points_1d; ++j)
          {
            higher.shape_values[i * n_q_points_1d + j] =
              nodal_basis_of_high[i].value(q_points[j]);
            higher.shape_gradients[i * n_q_points_1d + j] =
              nodal_basis_of_high[i].derivative().value(q_points[j]);
            higher.shape_hessians[i * n_q_points_1d + j] =
              nodal_basis_of_high[i].derivative().derivative().value(
                q_points[j]);
          }
        higher.shape_data_on_face[0][i] = nodal_basis_of_high[i].value(0);
        higher.shape_data_on_face[0][i + (higher.fe_degree + 1)] =
          nodal_basis_of_high[i].derivative().value(0);
        higher.shape_data_on_face[0][i + 2 * (higher.fe_degree + 1)] =
          nodal_basis_of_high[i].derivative().derivative().value(0);
        higher.shape_data_on_face[1][i] = nodal_basis_of_high[i].value(1);
        higher.shape_data_on_face[1][i + (higher.fe_degree + 1)] =
          nodal_basis_of_high[i].derivative().value(1);
        higher.shape_data_on_face[1][i + 2 * (higher.fe_degree + 1)] =
          nodal_basis_of_high[i].derivative().derivative().value(1);
      }

    /// NOTE as of 2021/03/06 the documentation of lexicographic_numbering is
    /// wrong: lexicographic_numbering defines a lexicographic-to-hierarchical
    /// mapping (l2h) and not vice versa as documented
    shape_info.lexicographic_numbering = inverse_lexicographic_transformation;
    shape_info.element_type            = raviart_thomas;
    shape_info.n_dimensions            = dim;
    shape_info.n_components            = dim;
    shape_info.dofs_per_component_on_cell =
      (higher.fe_degree + 1) *
      (dim > 1 ? Utilities::pow(lower.fe_degree + 1, dim - 1) : 1);
    // dofs_per_component_on_face cannot be set since different faces have
    // different numbers, Maximum number is std::pow(lower.fe_degree,dim-1)
    // other faces have 0
    shape_info.dofs_per_component_on_face = numbers::invalid_unsigned_int;
    shape_info.n_q_points                 = Utilities::pow(quad.size(), dim);
    shape_info.n_q_points_face =
      dim > 1 ? Utilities::pow(quad.size(), dim - 1) : 1;
    shape_info.data_access.reinit(dim, dim);
    for (std::size_t i = 0; i < dim; ++i)
      {
        for (std::size_t j = 0; j < dim; ++j)
          {
            shape_info.data_access(i, j) = i == j ? &higher : &lower;
          }
      }
  }

private:
  /**
   * convert lexicographic index to dealii normal index
   */
  unsigned int
  lexicographic_index_to_normal(unsigned int i);
  /**
   * convert deal-ii standard index to lexicographic index
   */
  unsigned int
  normal_index_to_lexicographic(unsigned int i);
  /**
   * compute one dimensional polynomial bases and lexicographic support points
   */
  void
  compute_tensor_product_basis(const unsigned int deg);

  /**
   * Compute transformation between lexicographic support points and normal
   * support points
   */
  void
  fill_lexicographic_numbering(const unsigned int deg);


  std::vector<unsigned int>
                            inverse_lexicographic_transformation; // lex to normal
  std::vector<unsigned int> lexicographic_transformation; // normal to lex
  std::vector<Polynomials::Polynomial<double>> nodal_basis_of_high;
  std::vector<Polynomials::Polynomial<double>> nodal_basis_of_low;
  std::vector<Point<dim>>                      lexicographic_support_points;

  /**
   * Only for internal use. Its full name is @p get_dofs_per_object_vector
   * function and it creates the @p dofs_per_object vector that is needed
   * within the constructor to be passed to the constructor of @p
   * FiniteElementData.
   */
  static std::vector<unsigned int>
  get_dpo_vector(const unsigned int degree);

  /**
   * Compute the vector used for the @p restriction_is_additive field passed
   * to the base class's constructor.
   */
  static std::vector<bool>
  get_ria_vector(const unsigned int degree);

  /**
   * This function returns @p true, if the shape function @p shape_index has
   * non-zero function values somewhere on the face @p face_index.
   *
   * Right now, this is only implemented for RT0 in 1D. Otherwise, returns
   * always @p true.
   */
  virtual bool
  has_support_on_face(const unsigned int shape_index,
                      const unsigned int face_index) const override;
  /**
   * Initialize the FiniteElement<dim>::generalized_support_points and
   * FiniteElement<dim>::generalized_face_support_points fields. Called from
   * the constructor.
   *
   * See the
   * @ref GlossGeneralizedSupport "glossary entry on generalized support points"
   * for more information.
   */
  void
  initialize_support_points(const unsigned int rt_degree);
};


/*@}*/


/// TODO fe_raviart_thomas_new.templates.h or alike
template <int dim>
template <typename Number>
void
FE_RaviartThomas_new<dim>::fill_shape_info(
  internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
  Quadrature<1>                                     quad) const
{
  using namespace internal::MatrixFreeFunctions;

  /// to be on the safe side set shape_info to its default
  shape_info = ShapeInfo<Number>{};

  AssertDimension(shape_info.data.size(), 0U); // double-check
  shape_info.data.reserve(2U);
  /// c++17: auto & high = shape_info.data.emplace_back();
  shape_info.data.emplace_back();
  shape_info.data.emplace_back();
  UnivariateShapeData<Number> &high = shape_info.data.front();
  UnivariateShapeData<Number> &low  = shape_info.data.back();

  low.element_type             = tensor_symmetric;
  low.fe_degree                = raw_polynomials_k.size() - 1;
  low.quadrature               = quad;
  low.n_q_points_1d            = quad.size();
  low.nodal_at_cell_boundaries = false;

  high.element_type             = tensor_symmetric;
  high.fe_degree                = raw_polynomials_kplus1.size() - 1;
  high.quadrature               = quad;
  high.n_q_points_1d            = quad.size();
  high.nodal_at_cell_boundaries = true;

  std::vector<double> q_points;
  std::transform(quad.get_points().cbegin(),
                 quad.get_points().cend(),
                 std::back_inserter(q_points),
                 [](const auto &point) { return point[0]; });

  const auto fill_shape_data =
    [&](UnivariateShapeData<Number> &                       info,
        const std::vector<Polynomials::Polynomial<double>> &raw_polynomials,
        const FullMatrix<double> &raw_to_shape_matrix) {
      /// size of data array: value + 1st derivative + 2nd derivative
      constexpr unsigned int nd = 3;

      const unsigned int n_dofs = info.fe_degree + 1;

      AssertDimension(q_points.size(), info.n_q_points_1d);

      info.shape_values    = AlignedVector<Number>(n_dofs * info.n_q_points_1d);
      info.shape_gradients = AlignedVector<Number>(n_dofs * info.n_q_points_1d);
      info.shape_hessians  = AlignedVector<Number>(n_dofs * info.n_q_points_1d);
      info.shape_data_on_face[0].resize(nd * n_dofs);
      info.shape_data_on_face[1].resize(nd * n_dofs);

      AssertDimension(n_dofs, raw_to_shape_matrix.m());                 // !!!
      AssertDimension(raw_polynomials.size(), raw_to_shape_matrix.n()); // !!!
      AssertDimension(n_dofs, raw_polynomials.size());

      std::vector<Tensor<1, nd, double>> raw_data;
      for (const auto &x_q : q_points)
        std::transform(raw_polynomials.cbegin(),
                       raw_polynomials.cend(),
                       std::back_inserter(raw_data),
                       [&](const auto &poly) {
                         Tensor<1, nd, double> values;
                         poly.value(x_q, nd - 1, values.begin_raw());
                         return values;
                       });

      AssertDimension(raw_data.size(), n_dofs * info.n_q_points_1d);

      std::vector<Tensor<1, nd, double>> raw_data_face;
      for (const auto &x_q : {0., 1.})
        std::transform(raw_polynomials.cbegin(),
                       raw_polynomials.cend(),
                       std::back_inserter(raw_data_face),
                       [&](const auto &poly) {
                         Tensor<1, nd, double> values;
                         poly.value(x_q, nd - 1, values.begin_raw());
                         return values;
                       });

      AssertDimension(raw_data_face.size(), 2 * n_dofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          for (unsigned int q = 0; q < info.n_q_points_1d; ++q)
            {
              const unsigned int stride         = q * n_dofs;
              const auto         raw_data_begin = raw_data.cbegin() + stride;

              Tensor<1, nd, double> shape_data;
              for (unsigned int j = 0; j < n_dofs; ++j)
                shape_data +=
                  raw_to_shape_matrix(i, j) * raw_data_begin[j]; // !!!

              info.shape_values[i * info.n_q_points_1d + q]    = shape_data[0];
              info.shape_gradients[i * info.n_q_points_1d + q] = shape_data[1];
              info.shape_hessians[i * info.n_q_points_1d + q]  = shape_data[2];
            }

          for (unsigned int face_no = 0; face_no < 2U; ++face_no)
            {
              const unsigned int stride = face_no * n_dofs;
              const auto raw_data_begin = raw_data_face.cbegin() + stride;

              Tensor<1, nd, double> shape_data;
              for (unsigned int j = 0; j < n_dofs; ++j)
                shape_data +=
                  raw_to_shape_matrix(i, j) * raw_data_begin[j]; // !!!

              info.shape_data_on_face[face_no][i]              = shape_data[0];
              info.shape_data_on_face[face_no][n_dofs + i]     = shape_data[1];
              info.shape_data_on_face[face_no][2 * n_dofs + i] = shape_data[2];
            }
        }
    };

  FullMatrix<double> raw_to_shape_k;
  raw_to_shape_k.copy_transposed(inverse_node_value_matrix_k);

  fill_shape_data(low, raw_polynomials_k, raw_to_shape_k);

  FullMatrix<double> raw_to_shape_kplus1;
  raw_to_shape_kplus1.copy_transposed(inverse_node_value_matrix_kplus1);

  fill_shape_data(high, raw_polynomials_kplus1, raw_to_shape_kplus1);

  /// NOTE as of 2021/03/06 the documentation of lexicographic_numbering is
  /// wrong: lexicographic_numbering defines a lexicographic-to-hierarchical
  /// mapping (l2h) and not vice versa as documented
  shape_info.lexicographic_numbering = this->l2h;
  shape_info.element_type            = raviart_thomas;
  shape_info.n_dimensions            = dim;
  shape_info.n_components            = dim;

  AssertDimension(this->n_dofs_per_cell() % dim, 0U);
  shape_info.dofs_per_component_on_cell = this->n_dofs_per_cell() / dim;
  /// dofs_per_component_on_face is not reasonable for this finite element: each
  /// vector component has faces with either 0 dofs or n_dofs_per_face(). we
  /// decide to set the maximum here
  shape_info.dofs_per_component_on_face = this->n_dofs_per_face();

  shape_info.n_q_points = Utilities::pow(quad.size(), dim);
  shape_info.n_q_points_face =
    dim > 1 ? Utilities::pow(quad.size(), dim - 1) : 1;

  AssertDimension(shape_info.data.size(), 2U);
  Assert(&high == &shape_info.data.front(), ExcInternalError());
  Assert(&low == &shape_info.data.back(), ExcInternalError());

  shape_info.data_access.reinit(dim, dim);
  for (auto dimension = 0U; dimension < dim; ++dimension)
    for (auto comp = 0U; comp < dim; ++comp)
      shape_info.data_access(dimension, comp) =
        dimension == comp ? &high : &low;
}


template <int dim>
inline double
FE_RaviartThomas_new<dim>::evaluate_node_functional_kplus1(
  const unsigned int                     i,
  const Polynomials::Polynomial<double> &poly,
  const Quadrature<1> &                  quad) const
{
  const unsigned int  k = this->degree - 1; // RT index
  std::vector<double> poly_values;
  /// nodal at endpoints 0 and 1
  if (i == 0U || i == (k + 1))
    {
      poly_values.emplace_back(i == 0U ? poly.value(0.) : poly.value(1.));
      return evaluate_node_functional_kplus1_impl<true>(
        i, make_array_view<double>(poly_values), quad);
    }
  /// moment-based for all remaining dofs
  std::transform(quad.get_points().cbegin(),
                 quad.get_points().cend(),
                 std::back_inserter(poly_values),
                 [&](const auto x_q) { return poly.value(x_q[0]); });
  return evaluate_node_functional_kplus1_impl<false>(
    i, make_array_view<double>(poly_values), quad);
}


template <int dim>
template <bool is_nodal>
inline double
FE_RaviartThomas_new<dim>::evaluate_node_functional_kplus1_impl(
  const unsigned int             i,
  const ArrayView<const double> &poly_values,
  const Quadrature<1> &          quad) const
{
  AssertIndexRange(i, node_polynomials_kminus1.size() + 2U);
  /// nodal at endpoints
  if (is_nodal)
    {
      AssertDimension(poly_values.size(), 1U);
      Assert(i == 0U || i == this->degree,
             ExcMessage("Only the first and last dof are nodal."));
      return poly_values[0];
    }

  /// k moments in the interior
  AssertDimension(poly_values.size(), quad.size());
  const auto &q_points  = quad.get_points();
  const auto &q_weights = quad.get_weights();
  const auto &node_poly = node_polynomials_kminus1[i - 1];
  double      eval      = 0.;
  for (auto q = 0U; q < quad.size(); ++q)
    {
      const double &x_q = q_points[q][0];
      eval += poly_values[q] * node_poly.value(x_q) * q_weights[q];
    }
  return eval;
}


template <int dim>
inline double
FE_RaviartThomas_new<dim>::evaluate_node_functional_k(
  const unsigned int                     i,
  const Polynomials::Polynomial<double> &poly,
  const Quadrature<1> &                  quad) const
{
  std::vector<double> poly_values;
  std::transform(quad.get_points().cbegin(),
                 quad.get_points().cend(),
                 std::back_inserter(poly_values),
                 [&](const auto x_q) { return poly.value(x_q[0]); });
  return evaluate_node_functional_k_impl(i,
                                         make_array_view<double>(poly_values),
                                         quad);
}


template <int dim>
inline double
FE_RaviartThomas_new<dim>::evaluate_node_functional_k_impl(
  const unsigned int             i,
  const ArrayView<const double> &poly_values,
  const Quadrature<1> &          quad) const
{
  AssertIndexRange(i, node_polynomials_k.size());
  AssertDimension(poly_values.size(), quad.size());
  const auto &q_points  = quad.get_points();
  const auto &q_weights = quad.get_weights();
  const auto &node_poly = node_polynomials_k[i];
  double      eval      = 0.;
  for (auto q = 0U; q < quad.size(); ++q)
    {
      const double &x_q = q_points[q][0];
      eval += poly_values[q] * node_poly.value(x_q) * q_weights[q];
    }
  return eval;
}


/* -------------- declaration of explicit specializations ------------- */

#ifndef DOXYGEN

template <>
void
FE_RaviartThomas_new<1>::initialize_restriction();

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
