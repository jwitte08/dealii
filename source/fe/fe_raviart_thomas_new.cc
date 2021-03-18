// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2020 by the deal.II authors
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


#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <iostream>
#include <memory>
#include <sstream>

// TODO: implement the adjust_quad_dof_index_for_face_orientation_table and
// adjust_line_dof_index_for_line_orientation_table fields, and write tests
// similar to bits/face_orientation_and_fe_q_*


DEAL_II_NAMESPACE_OPEN


namespace Tensors
{
  /**
   * Computes the Kronecker product of two matrices @p lhs and @p rhs. Matrix
   * types are forwarded and need to be a derivative of Table<2,Number>. Note
   * that the Kronecker product is not commutative.
   */
  template <typename MatrixTypeIn1,
            typename MatrixTypeIn2,
            typename MatrixTypeOut = std::decay_t<MatrixTypeIn1>>
  MatrixTypeOut
  kronecker_product(MatrixTypeIn1 &&lhs_in, MatrixTypeIn2 &&rhs_in)
  {
    auto &&lhs = std::forward<MatrixTypeIn1>(lhs_in);
    auto &&rhs = std::forward<MatrixTypeIn2>(rhs_in);

    const unsigned int n_rows_r = rhs.n_rows();
    const unsigned int n_cols_r = rhs.n_cols();
    const unsigned int n_rows_l = lhs.n_rows();
    const unsigned int n_cols_l = lhs.n_cols();

    MatrixTypeOut prod_matrix;
    prod_matrix.reinit(n_rows_l * n_rows_r, n_cols_l * n_cols_r);

    for (unsigned int il = 0; il < n_rows_l; ++il)
      for (unsigned int jl = 0; jl < n_cols_l; ++jl)
        for (unsigned int ir = 0; ir < n_rows_r; ++ir)
          for (unsigned int jr = 0; jr < n_cols_r; ++jr)
            prod_matrix(il * n_rows_r + ir, jl * n_cols_r + jr) =
              lhs(il, jl) * rhs(ir, jr);

    return prod_matrix;
  }


  /*
   * transforms an (anisotropic) multi-index into the canonical uni-index with
   * respect to lexicographical order. That is the first index of the
   * multi-index runs faster than second and so on.
   *
   * order : the order of the multi-index
   * sizes  : (anisotropic) size of each independent variable (mode)
   */
  template <int order, typename IntType = unsigned int>
  IntType
  multi_to_uniindex(const std::array<IntType, order> &multiindex,
                    const std::array<IntType, order> &sizes)
  {
    for (IntType k = 0; k < multiindex.size(); ++k)
      AssertIndexRange(multiindex[k], sizes[k]);
    IntType uniindex{0};
    for (int k = order - 1; k >= 0; --k)
      {
        // has no effect on purpose for k == order-1 (uniindex is zero)
        uniindex *= sizes[k];
        uniindex += multiindex[k];
      }
    const auto n_elem = std::accumulate(sizes.cbegin(),
                                        sizes.cend(),
                                        1,
                                        std::multiplies<IntType>());
    (void)n_elem;
    AssertIndexRange(uniindex, n_elem);

    return uniindex;
  }


  /*
   * transforms an (isotropic) multi-index into the canonical uni-index with
   * respect to lexicographical order. That is the first index of the
   * multi-index runs faster than second and so on.
   *
   * order : the order of the multi-index
   * size  : isotropic size of each index set (mode)
   */
  template <int order, typename IntType = unsigned int>
  IntType
  multi_to_uniindex(const std::array<IntType, order> &multiindex,
                    const IntType                     size)
  {
    std::array<IntType, order> sizes;
    sizes.fill(size);
    return multi_to_uniindex<order>(multiindex, sizes);
  }


  /*
   * transforms an uni-index into the canonical (anisotropic) multi-index with
   * respect to lexicographical order. That is the first index of the
   * multi-index runs faster than second and so on.
   *
   * order : the order of the multi-index
   * sizes : sizes of each independent variable (mode)
   */
  template <int order, typename IntType = unsigned int>
  std::array<IntType, order>
  uni_to_multiindex(IntType index, const std::array<IntType, order> &sizes)
  {
    const auto n_elem = std::accumulate(sizes.cbegin(),
                                        sizes.cend(),
                                        1,
                                        std::multiplies<IntType>());
    (void)n_elem;
    AssertIndexRange(index, n_elem);
    std::array<IntType, order> multiindex;
    for (int k = 0; k < order; ++k)
      {
        multiindex[k] = index % sizes[k];
        index         = index / sizes[k];
      }
    Assert(index == 0,
           ExcMessage("Uni-index has remainder after multi-index extraction."));
    for (IntType k = 0; k < multiindex.size(); ++k)
      AssertIndexRange(multiindex[k], sizes[k]);

    return multiindex;
  }


  /*
   * transforms an uni-index into the canonical (isotropic) multi-index with
   * respect to lexicographical order. That is the first index of the
   * multi-index runs faster than second and so on.
   *
   * order : the order of the multi-index
   * size  : isotropic size of each index set (mode)
   */
  template <int order, typename IntType = unsigned int>
  std::array<IntType, order>
  uni_to_multiindex(IntType index, const IntType size)
  {
    std::array<IntType, order> sizes;
    sizes.fill(size);
    return uni_to_multiindex<order>(index, sizes);
  }



  template <int order, typename IntType = unsigned int>
  struct TensorHelper
  {
    TensorHelper(const std::array<IntType, order> &sizes)
      : n(sizes)
    {}

    TensorHelper(const IntType size)
      : n([size]() {
        std::array<IntType, order> sizes;
        sizes.fill(size);
        return sizes;
      }())
    {}

    std::array<IntType, order>
    multi_index(const IntType index) const
    {
      return uni_to_multiindex<order, IntType>(index, n);
    }

    IntType
    uni_index(const std::array<IntType, order> &multi_index) const
    {
      return multi_to_uniindex<order, IntType>(multi_index, n);
    }

    std::vector<IntType>
    sliced_indices(const IntType index, const unsigned int mode) const
    {
      AssertThrow(order > 0, ExcMessage("Not implemented."));

      std::vector<IntType> indices;
      AssertIndexRange(mode, order);
      AssertIndexRange(index, size(mode));
      if (order == 1)
        {
          indices.emplace_back(index);
          return indices;
        }

      const auto restrict = [&](const std::array<IntType, order> &multiindex) {
        std::array<IntType, order - 1> slicedindex;
        for (auto m = 0U; m < mode; ++m)
          slicedindex[m] = multiindex[m];
        for (auto m = mode + 1; m < order; ++m)
          slicedindex[m - 1] = multiindex[m];
        return slicedindex;
      };
      const auto prolongate =
        [&](const std::array<IntType, order - 1> &slicedindex) {
          std::array<IntType, order> multiindex;
          for (auto m = 0U; m < mode; ++m)
            multiindex[m] = slicedindex[m];
          multiindex[mode] = index;
          for (auto m = mode + 1; m < order; ++m)
            multiindex[m] = slicedindex[m - 1];
          return multiindex;
        };

      TensorHelper<order - 1, IntType> slice(restrict(this->n));
      for (auto i = 0U; i < slice.n_flat(); ++i)
        {
          const auto sliced_index = slice.multi_index(i);
          const auto multi_index  = prolongate(sliced_index);
          indices.emplace_back(this->uni_index(multi_index));
        }
      return indices;
    }

    bool
    is_isotropic() const
    {
      for (auto direction = 1U; direction < order; ++direction)
        if (size(0U) != size(direction))
          return false;
      return true;
    }

    bool
    operator==(const TensorHelper<order, IntType> &other) const
    {
      return std::equal(n.cbegin(),
                        n.cend(),
                        other.n.cbegin(),
                        other.n.cend(),
                        [](const auto i, const auto j) { return i == j; });
    }

    IntType
    n_flat() const
    {
      return std::accumulate(n.cbegin(),
                             n.cend(),
                             static_cast<IntType>(1),
                             std::multiplies<IntType>());
    }

    IntType
    size(const unsigned int mode) const
    {
      AssertIndexRange(mode, order);
      return n[mode];
    }

    /**
     * Returns the minimum size of any dimension.
     */
    IntType
    min_size() const
    {
      IntType min = 0;
      for (const auto i : n)
        min = std::min(i, min);
      return min;
    }

    /**
     * Returns the maximum size of any dimension.
     */
    IntType
    max_size() const
    {
      IntType max = 0;
      for (const auto i : n)
        max = std::max(i, max);
      return max;
    }

    /**
     * Returns the dimension which has the minimum size of all dimensions. If
     * more than one dimension is of minimum size the first dimension with
     * minimum size is returned.
     */
    unsigned int
    min_dimension() const
    {
      return std::distance(n.cbegin(), std::min_element(n.cbegin(), n.cend()));
    }

    /**
     * Returns the dimension which has the maximum size of all dimensions. If
     * more than one dimension is of maximum size the first dimension with
     * maximum size is returned.
     */
    unsigned int
    max_dimension() const
    {
      return std::distance(n.cbegin(), std::max_element(n.cbegin(), n.cend()));
    }

    const std::array<IntType, order> &
    size() const
    {
      return n;
    }

    IntType
    collapsed_size_pre(const unsigned int direction) const
    {
      return std::accumulate(n.begin(),
                             n.begin() + direction,
                             1,
                             std::multiplies<IntType>{});
    }

    IntType
    collapsed_size_post(const unsigned int direction) const
    {
      return std::accumulate(n.begin() + direction + 1,
                             n.end(),
                             1,
                             std::multiplies<IntType>{});
    }

    const std::array<IntType, order> n;
  };
} // namespace Tensors


template <int dim>
FE_RaviartThomas_new<dim>::FE_RaviartThomas_new(const unsigned int deg)
  : FE_PolyTensor<dim>(
      PolynomialsRaviartThomas_new<dim>(deg),
      FiniteElementData<dim>(get_dpo_vector(deg),
                             dim,
                             deg + 1,
                             FiniteElementData<dim>::Hdiv),
      std::vector<bool>(PolynomialsRaviartThomas<dim>::n_polynomials(deg),
                        true),
      std::vector<ComponentMask>(PolynomialsRaviartThomas<dim>::n_polynomials(
                                   deg),
                                 std::vector<bool>(dim, true)))
  , raw_polynomials_kplus1(
      PolynomialsRaviartThomas_new<dim>::make_univariate_polynomials_high(deg))
  , raw_polynomials_k(
      PolynomialsRaviartThomas_new<dim>::make_univariate_polynomials_low(deg))
  , node_polynomials_k(Polynomials::Legendre::generate_complete_basis(deg))
  , node_polynomials_kminus1(
      deg > 0 ? Polynomials::Legendre::generate_complete_basis(deg - 1) :
                std::vector<Polynomials::Polynomial<double>>{})
{
  Assert(dim >= 2, ExcImpossibleInDim(dim));
  const unsigned int n_dofs = this->n_dofs_per_cell();

  this->mapping_kind = {mapping_raviart_thomas};

  /// Caching the hierarchical-to-lexicographic numbering and its inverse.
  h2l = std::move(make_hierarchical_to_lexicographic_index_map(deg));
  l2h = std::move(Utilities::invert_permutation(h2l));

  // First, initialize the
  // generalized support points and
  // quadrature weights, since they
  // are required for interpolation.
  initialize_support_points(deg);

  /// NEW compute node values from the tensor product of 1d node values and raw
  /// tensor product polynomials
  FullMatrix<double> node_value_matrix_kplus1;
  FullMatrix<double> node_value_matrix_k;
  {
    QGauss<1> quad(deg + 1);

    AssertDimension(raw_polynomials_kplus1.size(), deg + 2);
    node_value_matrix_kplus1.reinit(deg + 2, deg + 2);
    for (auto i = 0U; i < deg + 2; ++i)
      for (auto j = 0U; j < deg + 2; ++j)
        node_value_matrix_kplus1(i, j) =
          this->evaluate_node_functional_kplus1(i,
                                                raw_polynomials_kplus1[j],
                                                quad);

    AssertDimension(raw_polynomials_k.size(), deg + 1);
    node_value_matrix_k.reinit(deg + 1, deg + 1);
    for (auto i = 0U; i < deg + 1; ++i)
      for (auto j = 0U; j < deg + 1; ++j)
        node_value_matrix_k(i, j) =
          this->evaluate_node_functional_k(i, raw_polynomials_k[j], quad);
  }

  AssertDimension(node_value_matrix_kplus1.m(), node_value_matrix_kplus1.n());
  AssertDimension(node_value_matrix_k.m(), node_value_matrix_k.n());

  inverse_node_value_matrix_kplus1.reinit(node_value_matrix_kplus1.m(),
                                          node_value_matrix_kplus1.n());
  inverse_node_value_matrix_kplus1.invert(node_value_matrix_kplus1);

  inverse_node_value_matrix_k.reinit(node_value_matrix_k.m(),
                                     node_value_matrix_k.n());
  inverse_node_value_matrix_k.invert(node_value_matrix_k);

  FullMatrix<double> node_value_matrix(n_dofs, n_dofs);
  {
    /// dummy used to mimick node values for "artificial" dimensions
    FullMatrix<double> one(IdentityMatrix(1U));

    AssertDimension(n_dofs % dim, 0U);
    const unsigned int n_dofs_per_comp = n_dofs / dim;

    for (unsigned int comp = 0; comp < dim; ++comp)
      {
        const unsigned int offset_comp = comp * n_dofs_per_comp;

        const FullMatrix<double> &N_2 =
          dim > 2 ?
            (comp == 2 ? node_value_matrix_kplus1 : node_value_matrix_k) :
            one;
        const FullMatrix<double> &N_1 =
          dim > 1 ?
            (comp == 1 ? node_value_matrix_kplus1 : node_value_matrix_k) :
            one;
        const FullMatrix<double> &N_0 =
          dim > 0 ?
            (comp == 0 ? node_value_matrix_kplus1 : node_value_matrix_k) :
            one;

        const FullMatrix<double> &prod =
          Tensors::kronecker_product(N_2, Tensors::kronecker_product(N_1, N_0));

        for (auto i = 0U; i < n_dofs_per_comp; ++i)
          for (auto j = 0U; j < n_dofs_per_comp; ++j)
            node_value_matrix(l2h[offset_comp + i], offset_comp + j) =
              prod(i, j);
      }
  }

  // FullMatrix<double> node_value_matrix_hierarchical(node_value_matrix.m(),
  //                                                   node_value_matrix.n());
  // for (auto i = 0U; i < node_value_matrix.m(); ++i)
  //   for (auto j = 0U; j < node_value_matrix.n(); ++j)
  //     node_value_matrix_hierarchical(l2h[i], j) = node_value_matrix(i, j);

  this->inverse_node_matrix.reinit(n_dofs, n_dofs);
  this->inverse_node_matrix.invert(node_value_matrix);

  // From now on, the shape functions provided by FiniteElement::shape_value
  // and similar functions will be the correct ones, not
  // the raw shape functions from the polynomial space anymore.

  // Reinit the vectors of
  // restriction and prolongation
  // matrices to the right sizes.
  // Restriction only for isotropic
  // refinement
  this->reinit_restriction_and_prolongation_matrices(true);
  // Fill prolongation matrices with embedding operators
  FETools::compute_embedding_matrices(*this, this->prolongation);
  initialize_restriction();

  // TODO: the implementation makes the assumption that all faces have the
  // same number of dofs
  AssertDimension(this->n_unique_faces(), 1);
  const unsigned int face_no = 0;

  // TODO[TL]: for anisotropic refinement we will probably need a table of
  // submatrices with an array for each refine case
  FullMatrix<double> face_embeddings[GeometryInfo<dim>::max_children_per_face];
  for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_face; ++i)
    face_embeddings[i].reinit(this->n_dofs_per_face(face_no),
                              this->n_dofs_per_face(face_no));
  FETools::compute_face_embedding_matrices<dim, double>(*this,
                                                        face_embeddings,
                                                        0,
                                                        0);
  this->interface_constraints.reinit((1 << (dim - 1)) *
                                       this->n_dofs_per_face(face_no),
                                     this->n_dofs_per_face(face_no));
  unsigned int target_row = 0;
  for (unsigned int d = 0; d < GeometryInfo<dim>::max_children_per_face; ++d)
    for (unsigned int i = 0; i < face_embeddings[d].m(); ++i)
      {
        for (unsigned int j = 0; j < face_embeddings[d].n(); ++j)
          this->interface_constraints(target_row, j) = face_embeddings[d](i, j);
        ++target_row;
      }
}


template <int dim>
std::vector<unsigned int>
FE_RaviartThomas_new<dim>::make_hierarchical_to_lexicographic_index_map(
  const unsigned int fe_degree) const
{
  const unsigned int n_dofs_per_comp = this->n_dofs_per_cell() / dim;

  std::vector<unsigned int> h2l;
  h2l.reserve(n_dofs_per_comp * dim);

  /// In hierarchical numbering dofs on faces come first
  for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
    {
      const unsigned int stride_x = face_no < 2 ? fe_degree + 2 : 1;
      const unsigned int stride_y =
        face_no < 4 ? (fe_degree + 2) * (fe_degree + 1) : fe_degree + 1;
      const unsigned int offset =
        (face_no % 2) * Utilities::pow(fe_degree + 1, 1 + face_no / 2);
      for (unsigned int j = 0; j < (dim > 2 ? fe_degree + 1 : 1); ++j)
        for (unsigned int i = 0; i < fe_degree + 1; ++i)
          h2l.push_back((face_no / 2) * n_dofs_per_comp + offset +
                        i * stride_x + j * stride_y);
    }

  /// In hierarchical numbering dofs associated with the interior of a cell
  /// comes second and last.
  for (unsigned int k = 0; k < (dim > 2 ? fe_degree + 1 : 1); ++k)
    for (unsigned int j = 0; j < (dim > 1 ? fe_degree + 1 : 1); ++j)
      for (unsigned int i = 1; i < fe_degree + 1; ++i)
        h2l.push_back(k * (fe_degree + 1) * (fe_degree + 2) +
                      j * (fe_degree + 2) + i);
  if (dim > 1)
    for (unsigned int k = 0; k < (dim > 2 ? fe_degree + 1 : 1); ++k)
      for (unsigned int j = 1; j < fe_degree + 1; ++j)
        for (unsigned int i = 0; i < fe_degree + 1; ++i)
          h2l.push_back(n_dofs_per_comp +
                        k * (fe_degree + 1) * (fe_degree + 2) +
                        j * (fe_degree + 1) + i);
  if (dim > 2)
    for (unsigned int k = 1; k < fe_degree + 1; ++k)
      for (unsigned int j = 0; j < fe_degree + 1; ++j)
        for (unsigned int i = 0; i < fe_degree + 1; ++i)
          h2l.push_back(2 * n_dofs_per_comp +
                        k * (fe_degree + 1) * (fe_degree + 1) +
                        j * (fe_degree + 1) + i);

  AssertDimension(h2l.size(), this->dofs_per_cell);

#ifdef DEBUG
  // assert that we have a valid permutation
  std::vector<unsigned int> copy(h2l);
  std::sort(copy.begin(), copy.end());
  for (unsigned int i = 0; i < copy.size(); ++i)
    AssertDimension(i, copy[i]);
#endif

  return h2l;
}


template <int dim>
std::string
FE_RaviartThomas_new<dim>::get_name() const
{
  // note that the
  // FETools::get_fe_by_name
  // function depends on the
  // particular format of the string
  // this function returns, so they
  // have to be kept in synch

  // note that this->degree is the maximal
  // polynomial degree and is thus one higher
  // than the argument given to the
  // constructor
  std::ostringstream namebuf;
  namebuf << "FE_RaviartThomas_new<" << dim << ">(" << this->degree - 1 << ")";

  return namebuf.str();
}


template <int dim>
std::unique_ptr<FiniteElement<dim, dim>>
FE_RaviartThomas_new<dim>::clone() const
{
  return std::make_unique<FE_RaviartThomas_new<dim>>(*this);
}


//---------------------------------------------------------------------------
// Auxiliary and internal functions
//---------------------------------------------------------------------------


template <int dim>
void
FE_RaviartThomas_new<dim>::initialize_support_points(const unsigned int deg)
{
  QGauss<dim>        cell_quadrature(deg + 1);
  const unsigned int n_interior_points = (deg > 0) ? cell_quadrature.size() : 0;

  // TODO: the implementation makes the assumption that all faces have the
  // same number of dofs
  AssertDimension(this->n_unique_faces(), 1);
  const unsigned int face_no = 0;

  unsigned int n_face_points = (dim > 1) ? 1 : 0;
  // compute (deg+1)^(dim-1)
  for (unsigned int d = 1; d < dim; ++d)
    n_face_points *= deg + 1;


  this->generalized_support_points.resize(
    GeometryInfo<dim>::faces_per_cell * n_face_points + n_interior_points);
  this->generalized_face_support_points[face_no].resize(n_face_points);

  // Number of the point being entered
  unsigned int current = 0;

  if (dim > 1)
    {
      QGauss<dim - 1>                   face_points(deg + 1);
      TensorProductPolynomials<dim - 1> legendre =
        Polynomials::Legendre::generate_complete_basis(deg);

      boundary_weights.reinit(n_face_points, legendre.n());

      for (unsigned int k = 0; k < n_face_points; ++k)
        {
          this->generalized_face_support_points[face_no][k] =
            face_points.point(k);
          // Compute its quadrature
          // contribution for each
          // moment.
          for (unsigned int i = 0; i < legendre.n(); ++i)
            {
              boundary_weights(k, i) =
                face_points.weight(k) *
                legendre.compute_value(i, face_points.point(k));
            }
        }

      Quadrature<dim> faces =
        QProjector<dim>::project_to_all_faces(this->reference_cell_type(),
                                              face_points);
      for (; current < GeometryInfo<dim>::faces_per_cell * n_face_points;
           ++current)
        {
          // Enter the support point
          // into the vector
          this->generalized_support_points[current] =
            faces.point(current + QProjector<dim>::DataSetDescriptor::face(
                                    this->reference_cell_type(),
                                    0,
                                    true,
                                    false,
                                    false,
                                    n_face_points));
        }
    }

  if (deg == 0)
    return;

  // Create Legendre basis for the space D_xi Q_k
  std::unique_ptr<AnisotropicPolynomials<dim>> polynomials[dim];
  for (unsigned int dd = 0; dd < dim; ++dd)
    {
      std::vector<std::vector<Polynomials::Polynomial<double>>> poly(dim);
      for (unsigned int d = 0; d < dim; ++d)
        poly[d] = Polynomials::Legendre::generate_complete_basis(deg);
      poly[dd] = Polynomials::Legendre::generate_complete_basis(deg - 1);

      polynomials[dd] = std::make_unique<AnisotropicPolynomials<dim>>(poly);
    }

  interior_weights.reinit(
    TableIndices<3>(n_interior_points, polynomials[0]->n(), dim));

  for (unsigned int k = 0; k < cell_quadrature.size(); ++k)
    {
      this->generalized_support_points[current++] = cell_quadrature.point(k);
      for (unsigned int i = 0; i < polynomials[0]->n(); ++i)
        for (unsigned int d = 0; d < dim; ++d)
          interior_weights(k, i, d) =
            cell_quadrature.weight(k) *
            polynomials[d]->compute_value(i, cell_quadrature.point(k));
    }

  Assert(current == this->generalized_support_points.size(),
         ExcInternalError());
}



template <>
void
FE_RaviartThomas_new<1>::initialize_restriction()
{
  // there is only one refinement case in 1d,
  // which is the isotropic one (first index of
  // the matrix array has to be 0)
  for (unsigned int i = 0; i < GeometryInfo<1>::max_children_per_cell; ++i)
    this->restriction[0][i].reinit(0, 0);
}



// This function is the same Raviart-Thomas interpolation performed by
// interpolate. Still, we cannot use interpolate, since it was written
// for smooth functions. The functions interpolated here are not
// smooth, maybe even not continuous. Therefore, we must double the
// number of quadrature points in each direction in order to integrate
// only smooth functions.

// Then again, the interpolated function is chosen such that the
// moments coincide with the function to be interpolated.

template <int dim>
void
FE_RaviartThomas_new<dim>::initialize_restriction()
{
  using namespace internal::MatrixFreeFunctions;

  Assert(dim > 1, ExcMessage("dim == 1 needs to be treated as special case."));

  const unsigned int iso = RefinementCase<dim>::isotropic_refinement - 1;
  AssertIndexRange(iso, this->restriction.size());

  const unsigned int k = this->degree - 1; // RT_k

  const unsigned n_q_points_1d = k + 1;

  QGauss<1> unit_quad(n_q_points_1d);

  ShapeInfo<double> shape_info;
  this->fill_shape_info(shape_info, unit_quad);

  const unsigned int   n_dofs_per_vertex_kplus1 = 1U;
  const unsigned int   n_dofs_per_line_kplus1   = k + 2;
  FiniteElementData<1> fe_data_kplus1({n_dofs_per_vertex_kplus1,
                                       n_dofs_per_line_kplus1},
                                      1,
                                      k + 1,
                                      FiniteElementData<1>::H1);

  std::vector<FullMatrix<double>> restriction_matrices_kplus1;

  /// Assembling univariate restriction matrices for degree (k+1)
  {
    const auto &shape_data_kplus1 = shape_info.get_shape_data(0, 0);

    AssertDimension(shape_data_kplus1.fe_degree, k + 1); // double-check
    AssertDimension(shape_data_kplus1.n_q_points_1d,
                    n_q_points_1d); // double-check

    const unsigned int n_dofs_fine    = shape_data_kplus1.fe_degree + 1;
    const unsigned int n_nodes_coarse = n_dofs_fine;

    const auto evaluate_node_functional_kplus1 =
      [&](const unsigned int   i,
          const unsigned int   j,
          const Quadrature<1> &quad) {
        AssertIndexRange(j, n_dofs_fine);
        ArrayView<const double> view_shape_values;
        /// nodal at endpoints 0 and 1 ...
        if (i == 0U || i == (k + 1))
          {
            const unsigned int face_no = i == 0U ? 0 : 1;
            view_shape_values.reinit(
              shape_data_kplus1.shape_data_on_face[face_no].begin() + j, 1U);
            return evaluate_node_functional_kplus1_impl<true>(i,
                                                              view_shape_values,
                                                              quad);
          }
        /// ... moment-based for all remaining dofs
        view_shape_values.reinit(shape_data_kplus1.shape_values.begin() +
                                   j * n_q_points_1d,
                                 n_q_points_1d);
        return evaluate_node_functional_kplus1_impl<false>(i,
                                                           view_shape_values,
                                                           quad);
      };

    for (auto child = 0U; child < GeometryInfo<1>::max_children_per_cell;
         ++child)
      {
        FullMatrix<double> &restriction_matrix_kplus1 =
          restriction_matrices_kplus1.emplace_back(n_nodes_coarse, n_dofs_fine);

        Quadrature<1> child_quad =
          QProjector<1>::project_to_child(fe_data_kplus1.reference_cell_type(),
                                          unit_quad,
                                          child);

        for (unsigned int j = 0; j < n_dofs_fine; ++j)
          for (unsigned int i = 0; i < n_nodes_coarse; ++i)
            restriction_matrix_kplus1(i, j) =
              evaluate_node_functional_kplus1(i, j, child_quad);
      }
  }

  std::vector<FullMatrix<double>> restriction_matrices_k;

  /// Assembling univariate restriction matrices for degree k
  {
    Assert(dim > 1, ExcInternalError());
    const auto &shape_data_k = shape_info.get_shape_data(1, 0);

    AssertDimension(shape_data_k.fe_degree, k);
    AssertDimension(shape_data_k.n_q_points_1d, n_q_points_1d);

    const unsigned int n_dofs_fine    = shape_data_k.fe_degree + 1;
    const unsigned int n_nodes_coarse = n_dofs_fine;

    const auto evaluate_node_functional_k = [&](const unsigned int   i,
                                                const unsigned int   j,
                                                const Quadrature<1> &quad) {
      AssertIndexRange(j, n_dofs_fine);
      ArrayView<const double> view_shape_values;
      view_shape_values.reinit(shape_data_k.shape_values.begin() +
                                 j * n_q_points_1d,
                               n_q_points_1d);
      return evaluate_node_functional_k_impl(i, view_shape_values, quad);
    };

    for (auto child = 0U; child < GeometryInfo<1>::max_children_per_cell;
         ++child)
      {
        FullMatrix<double> &restriction_matrix_k =
          restriction_matrices_k.emplace_back(n_nodes_coarse, n_dofs_fine);

        Quadrature<1> child_quad =
          QProjector<1>::project_to_child(fe_data_kplus1.reference_cell_type(),
                                          unit_quad,
                                          child);

        for (unsigned int j = 0; j < n_dofs_fine; ++j)
          for (unsigned int i = 0; i < n_nodes_coarse; ++i)
            restriction_matrix_k(i, j) =
              evaluate_node_functional_k(i, j, child_quad);
      }
  }

  Tensors::TensorHelper<dim, unsigned int> th_children(
    GeometryInfo<1>::max_children_per_cell);

  AssertDimension(this->restriction[iso].size(), th_children.n_flat());

  const unsigned int n_dofs_per_comp = this->n_dofs_per_cell() / dim;

  /// dummy used to mimick a restriction matrix for artificial dimensions
  FullMatrix<double> one(IdentityMatrix(1U));

  /// NOTE we want to evaluate the part of the node functionals belonging to
  /// this child cell in all fine-grid shape functions. node functionals are
  /// defined on the parent cell (i.e. the once refined unit cell), which
  /// includes functionals associated with all parent facets. if we
  /// now loop over all children to evaluate the moment-based functionals
  /// restricted to each child, we have to take care that we do not add
  /// contributions of child facets that are in the interior of the parent cell:
  /// we prevent this by determing the univariate node function index associated
  /// with these facets and zero them out ("purge").

  /// TODO treat non-standard face orientations ?!
  const auto purged_restriction_matrix_kplus1 =
    [&](const unsigned int child_1d) {
      const auto &       src = restriction_matrices_kplus1[child_1d];
      FullMatrix<double> dst = src;
      const unsigned int nonparent_face_no_1d = child_1d == 0 ? 1 : 0;
      const unsigned int node_index_1d = nonparent_face_no_1d == 0 ? 0 : k + 1;
      for (auto j = 0U; j < dst.n(); ++j)
        dst(node_index_1d, j) = 0.;
      return dst;
    };

  for (auto child = 0U; child < GeometryInfo<dim>::max_children_per_cell;
       ++child)
    {
      auto &restriction_matrix_child = this->restriction[iso][child];

      AssertDimension(restriction_matrix_child.m(), this->n_dofs_per_cell());
      AssertDimension(restriction_matrix_child.n(), this->n_dofs_per_cell());

      const auto mchild = th_children.multi_index(child);

      for (auto comp = 0U; comp < dim; ++comp)
        {
          const unsigned int offset_comp = comp * n_dofs_per_comp;

          const FullMatrix<double> &R_2 =
            dim > 2 ? comp == 2 ? purged_restriction_matrix_kplus1(mchild[2]) :
                                  restriction_matrices_k[mchild[2]] :
                      one;
          const FullMatrix<double> &R_1 =
            dim > 1 ? comp == 1 ? purged_restriction_matrix_kplus1(mchild[1]) :
                                  restriction_matrices_k[mchild[1]] :
                      one;
          const FullMatrix<double> &R_0 =
            dim > 0 ? comp == 0 ? purged_restriction_matrix_kplus1(mchild[0]) :
                                  restriction_matrices_k[mchild[0]] :
                      one;

          const FullMatrix<double> &prod =
            Tensors::kronecker_product(R_2,
                                       Tensors::kronecker_product(R_1, R_0));

          for (auto i = 0U; i < n_dofs_per_comp; ++i)
            for (auto j = 0U; j < n_dofs_per_comp; ++j)
              restriction_matrix_child(l2h[offset_comp + i],
                                       l2h[offset_comp + j]) = prod(i, j);
        }
    }
}



template <int dim>
std::vector<unsigned int>
FE_RaviartThomas_new<dim>::get_dpo_vector(const unsigned int deg)
{
  // the element is face-based and we have
  // (deg+1)^(dim-1) DoFs per face
  unsigned int dofs_per_face = 1;
  for (unsigned int d = 1; d < dim; ++d)
    dofs_per_face *= deg + 1;

  // and then there are interior dofs
  const unsigned int interior_dofs = dim * deg * dofs_per_face;

  std::vector<unsigned int> dpo(dim + 1);
  dpo[dim - 1] = dofs_per_face;
  dpo[dim]     = interior_dofs;

  return dpo;
}



template <int dim>
std::pair<Table<2, bool>, std::vector<unsigned int>>
FE_RaviartThomas_new<dim>::get_constant_modes() const
{
  Table<2, bool> constant_modes(dim, this->n_dofs_per_cell());
  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int i = 0; i < this->n_dofs_per_cell(); ++i)
      constant_modes(d, i) = true;
  std::vector<unsigned int> components;
  for (unsigned int d = 0; d < dim; ++d)
    components.push_back(d);
  return std::pair<Table<2, bool>, std::vector<unsigned int>>(constant_modes,
                                                              components);
}



//---------------------------------------------------------------------------
// Data field initialization
//---------------------------------------------------------------------------


template <int dim>
bool
FE_RaviartThomas_new<dim>::has_support_on_face(
  const unsigned int shape_index,
  const unsigned int face_index) const
{
  AssertIndexRange(shape_index, this->n_dofs_per_cell());
  AssertIndexRange(face_index, GeometryInfo<dim>::faces_per_cell);

  // Return computed values if we
  // know them easily. Otherwise, it
  // is always safe to return true.
  switch (this->degree)
    {
      case 1:
        {
          switch (dim)
            {
              case 2:
                {
                  // only on the one
                  // non-adjacent face
                  // are the values
                  // actually zero. list
                  // these in a table
                  return (face_index !=
                          GeometryInfo<dim>::opposite_face[shape_index]);
                }

              default:
                return true;
            }
        }

      default: // other rt_order
        return true;
    }

  return true;
}



// template <int dim>
// void
// FE_RaviartThomas_new<dim>::
//   convert_generalized_support_point_values_to_dof_values(
//     const std::vector<Vector<double>> &support_point_values,
//     std::vector<double> &              nodal_values) const
// {
//   Assert(support_point_values.size() ==
//   this->generalized_support_points.size(),
//          ExcDimensionMismatch(support_point_values.size(),
//                               this->generalized_support_points.size()));
//   Assert(nodal_values.size() == this->n_dofs_per_cell(),
//          ExcDimensionMismatch(nodal_values.size(), this->n_dofs_per_cell()));
//   Assert(support_point_values[0].size() == this->n_components(),
//          ExcDimensionMismatch(support_point_values[0].size(),
//                               this->n_components()));

//   std::fill(nodal_values.begin(), nodal_values.end(), 0.);

//   const unsigned int n_face_points = boundary_weights.size(0);
//   for (unsigned int face : GeometryInfo<dim>::face_indices())
//     for (unsigned int k = 0; k < n_face_points; ++k)
//       for (unsigned int i = 0; i < boundary_weights.size(1); ++i)
//         {
//           nodal_values[i + face * this->n_dofs_per_face(face)] +=
//             boundary_weights(k, i) *
//             support_point_values[face * n_face_points + k](
//               GeometryInfo<dim>::unit_normal_direction[face]);
//         }

//   // TODO: the implementation makes the assumption that all faces have the
//   // same number of dofs
//   AssertDimension(this->n_unique_faces(), 1);
//   const unsigned int face_no = 0;

//   const unsigned int start_cell_dofs =
//     GeometryInfo<dim>::faces_per_cell * this->n_dofs_per_face(face_no);
//   const unsigned int start_cell_points =
//     GeometryInfo<dim>::faces_per_cell * n_face_points;

//   const unsigned int n_interior_nodes_per_component =
//   interior_weights.size(1); for (unsigned int k = 0; k <
//   interior_weights.size(0); ++k)
//     for (unsigned int d = 0; d < dim; ++d)
//       for (unsigned int i = 0; i < n_interior_nodes_per_component; ++i)
//         nodal_values[start_cell_dofs + d * n_interior_nodes_per_component +
//                      i] += interior_weights(k, i, d) *
//                            support_point_values[k + start_cell_points](d);
//   /// OLD
//   // for (unsigned int k = 0; k < interior_weights.size(0); ++k)
//   //   for (unsigned int i = 0; i < interior_weights.size(1); ++i)
//   //     for (unsigned int d = 0; d < dim; ++d)
//   // nodal_values[start_cell_dofs + i * dim + d] +=
//   //   interior_weights(k, i, d) *
//   //   support_point_values[k + start_cell_points](d);
// }


template <int dim>
std::size_t
FE_RaviartThomas_new<dim>::memory_consumption() const
{
  Assert(false, ExcNotImplemented());
  return 0;
}



// explicit instantiations
#include "fe_raviart_thomas_new.inst"


DEAL_II_NAMESPACE_CLOSE
