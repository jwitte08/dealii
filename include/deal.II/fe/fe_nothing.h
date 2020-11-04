// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2019 by the deal.II authors
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

#ifndef dealii_fe_nothing_h
#define dealii_fe_nothing_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe.h>

DEAL_II_NAMESPACE_OPEN


/*!@addtogroup fe */
/*@{*/

/**
 * Definition of a finite element space with zero degrees of freedom and that,
 * consequently, can only represent a single function: the zero function.
 * This class is
 * useful (in the context of an hp method) to represent empty cells in the
 * triangulation on which no degrees of freedom should be allocated, or to
 * describe a field that is extended by zero to a part of the domain where we
 * don't need it.  Thus a triangulation may be divided into two regions: an
 * active region where normal elements are used, and an inactive region where
 * FE_Nothing elements are used.  The hp::DoFHandler will therefore assign no
 * degrees of freedom to the FE_Nothing cells, and this subregion is therefore
 * implicitly deleted from the computation. step-10 and step-46 show use cases
 * for this element. An interesting application for this element is also
 * presented in the paper @cite Cangiani2012.
 *
 * Note that some care must be taken that the resulting mesh topology
 * continues to make sense when FE_Nothing elements are introduced. This is
 * particularly true when dealing with hanging node constraints, because the
 * library makes some basic assumptions about the nature of those constraints.
 * The following geometries are acceptable:
 * @code
 * +---------+----+----+
 * |         | 0  |    |
 * |    1    +----+----+
 * |         | 0  |    |
 * +---------+----+----+
 * @endcode
 * @code
 * +---------+----+----+
 * |         | 1  |    |
 * |    0    +----+----+
 * |         | 1  |    |
 * +---------+----+----+
 * @endcode
 * Here, 0 denotes an FE_Nothing cell, and 1 denotes some other element type.
 * The library has no difficulty computing the necessary hanging node
 * constraints in these cases (i.e. no constraint). However, the following
 * geometry is NOT acceptable (at least in the current implementation):
 * @code
 * +---------+----+----+
 * |         | 0  |    |
 * |    1    +----+----+
 * |         | 1  |    |
 * +---------+----+----+
 * @endcode
 * The distinction lies in the mixed nature of the child faces, a case we have
 * not implemented as of yet.
 */
template <int dim, int spacedim = dim>
class FE_Nothing : public FiniteElement<dim, spacedim>
{
public:
  /**
   * Constructor. First argument denotes the number of components to give this
   * finite element (default = 1).
   *
   * Second argument decides whether FE_Nothing will dominate any other FE in
   * compare_for_domination() (default = false). Therefore at interfaces where,
   * for example, a Q1 meets an FE_Nothing, we will force the traces of the two
   * functions to be the same. Because the FE_Nothing encodes a space that is
   * zero everywhere, this means that the Q1 field will be forced to become zero
   * at this interface.
   */
  FE_Nothing(const unsigned int n_components = 1, const bool dominate = false);

  virtual std::unique_ptr<FiniteElement<dim, spacedim>>
  clone() const override;

  /**
   * Return a string that uniquely identifies a finite element. In this case
   * it is <code>FE_Nothing@<dim@></code>.
   */
  virtual std::string
  get_name() const override;

  // for documentation, see the FiniteElement base class
  virtual UpdateFlags
  requires_update_flags(const UpdateFlags update_flags) const override;

  /**
   * Return the value of the @p ith shape function at the point @p p. @p p is
   * a point on the reference element. Because the current element has no
   * degrees of freedom, this function should obviously not be called in
   * practice.  All this function really does, therefore, is trigger an
   * exception.
   */
  virtual double
  shape_value(const unsigned int i, const Point<dim> &p) const override;

  virtual void
  fill_fe_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const CellSimilarity::Similarity                            cell_similarity,
    const Quadrature<dim> &                                     quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual void
  fill_fe_face_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual void
  fill_fe_subface_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const unsigned int                                          sub_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  /**
   * Prepare internal data structures and fill in values independent of the
   * cell. Returns a pointer to an object of which the caller of this function
   * then has to assume ownership (which includes destruction when it is no
   * more needed).
   *
   * In the current case, this function just returns a default pointer, since
   * no meaningful data exists for this element.
   */
  virtual std::unique_ptr<
    typename FiniteElement<dim, spacedim>::InternalDataBase>
  get_data(
    const UpdateFlags             update_flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim> &       quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  /**
   * @copydoc FiniteElement::compare_for_domination()
   *
   * In the current case, this element is assumed to dominate if the second
   * argument in the constructor @p dominate is true. When this argument is
   * false and @p fe_other is also of type FE_Nothing(), either element can
   * dominate. Otherwise there are no_requirements.
   */
  virtual FiniteElementDomination::Domination
  compare_for_domination(const FiniteElement<dim, spacedim> &fe_other,
                         const unsigned int codim = 0) const override final;



  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_vertex_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_line_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_quad_dof_identities(const FiniteElement<dim, spacedim> &fe_other,
                         const unsigned int face_no = 0) const override;

  virtual bool
  hp_constraints_are_implemented() const override;

  /**
   * Return the matrix interpolating from the given finite element to the
   * present one. Since the current finite element has no degrees of freedom,
   * the interpolation matrix is necessarily empty.
   */
  virtual void
  get_interpolation_matrix(
    const FiniteElement<dim, spacedim> &source_fe,
    FullMatrix<double> &                interpolation_matrix) const override;

  /**
   * Return the matrix interpolating from a face of one element to the face
   * of the neighboring element. The size of the matrix is then
   * <tt>source.#dofs_per_face</tt> times <tt>this->#dofs_per_face</tt>.
   *
   * Since the current finite element has no degrees of freedom, the
   * interpolation matrix is necessarily empty.
   */

  virtual void
  get_face_interpolation_matrix(const FiniteElement<dim, spacedim> &source_fe,
                                FullMatrix<double> &interpolation_matrix,
                                const unsigned int  face_no = 0) const override;


  /**
   * Return the matrix interpolating from a face of one element to the
   * subface of the neighboring element. The size of the matrix is then
   * <tt>source.#dofs_per_face</tt> times <tt>this->#dofs_per_face</tt>.
   *
   * Since the current finite element has no degrees of freedom, the
   * interpolation matrix is necessarily empty.
   */

  virtual void
  get_subface_interpolation_matrix(
    const FiniteElement<dim, spacedim> &source_fe,
    const unsigned int                  index,
    FullMatrix<double> &                interpolation_matrix,
    const unsigned int                  face_no = 0) const override;

  /**
   * @return true if the FE dominates any other.
   */
  bool
  is_dominating() const;

  /**
   * Comparison operator. In addition to the fields already checked by
   * FiniteElement::operator==(), this operator also checks for equality
   * of the arguments passed to the constructors of the current object
   * as well as the object against which the comparison is done (which
   * for this purpose obviously also needs to be of type FE_Nothing).
   */
  virtual bool
  operator==(const FiniteElement<dim, spacedim> &fe) const override;

private:
  /**
   * If true, this element will dominate any other apart from itself in
   * compare_for_domination(). This is because a space that only contains the
   * zero function is definitely smaller (and consequently dominant) when
   * compared to any other finite element space.
   */
  const bool dominate;
};


/*@}*/

DEAL_II_NAMESPACE_CLOSE

#endif
