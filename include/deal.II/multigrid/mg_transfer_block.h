// ---------------------------------------------------------------------
//
// Copyright (C) 2001 - 2020 by the deal.II authors
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

#ifndef dealii_mg_transfer_block_h
#define dealii_mg_transfer_block_h

#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <memory>


DEAL_II_NAMESPACE_OPEN


// Forward declaration
#ifndef DOXYGEN
template <int dim, int spacedim>
class DoFHandler;
#endif

/*
 * MGTransferBase is defined in mg_base.h
 */

/*!@addtogroup mg */
/*@{*/

/**
 * Implementation of matrix generation for MGTransferBlock.
 *
 * This is the base class for MGTransfer objects for systems of equations
 * where multigrid is applied only to one or some blocks, where a
 * @ref GlossBlock
 * comprises all degrees of freedom generated by one base element.
 *
 * @author Guido Kanschat, 2001-2003
 */
class MGTransferBlockBase
{
public:
  /**
   * Constructor without constraint matrices. Use this constructor only with
   * discontinuous finite elements or with no local refinement.
   */
  MGTransferBlockBase();

  /**
   * Constructor with constraint matrices as well as mg_constrained_dofs.
   */
  MGTransferBlockBase(const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Memory used by this object.
   */
  std::size_t
  memory_consumption() const;

protected:
  /**
   * Actually build the prolongation matrices for each level.
   *
   * This function is only called by derived classes. These can also set the
   * member variables #selected and others to restrict the transfer matrices
   * to certain blocks.
   */
  template <int dim, int spacedim>
  void
  build(const DoFHandler<dim, spacedim> &dof_handler);

  /**
   * Actually build the prolongation matrices for each level.
   *
   * This function is only called by derived classes. These can also set the
   * member variables #selected and others to restrict the transfer matrices
   * to certain blocks.
   *
   * @deprecated Use build() instead.
   */
  template <int dim, int spacedim>
  DEAL_II_DEPRECATED void
  build_matrices(const DoFHandler<dim, spacedim> &dof,
                 const DoFHandler<dim, spacedim> &mg_dof);

  /**
   * Flag of selected blocks.
   *
   * The transfer operators only act on the blocks having a <tt>true</tt>
   * entry here.
   */
  // TODO: rename this to block_mask, in the same way as has already been done
  // in MGTransferComponent, and give it type BlockMask
  std::vector<bool> selected;

  /**
   * Number of blocks of multigrid vector.
   */
  unsigned int n_mg_blocks;

  /**
   * For each block of the whole block vector, list to what block of the
   * multigrid vector it is mapped. Since depending on #selected, there may be
   * fewer multilevel blocks than original blocks, some of the entries may be
   * illegal unsigned integers.
   */
  // TODO: rename this to mg_block_mask, in the same way as has already been
  // done in MGTransferComponent, and give it type BlockMask
  std::vector<unsigned int> mg_block;

  /**
   * Sizes of the multi-level vectors.
   */
  mutable std::vector<std::vector<types::global_dof_index>> sizes;

  /**
   * Start index of each block.
   */
  std::vector<types::global_dof_index> block_start;

  /**
   * Start index of each block on all levels.
   */
  std::vector<std::vector<types::global_dof_index>> mg_block_start;

  /**
   * Call build() function first.
   */
  DeclException0(ExcMatricesNotBuilt);

private:
  std::vector<std::shared_ptr<BlockSparsityPattern>> prolongation_sparsities;

protected:
  /**
   * The actual prolongation matrix. column indices belong to the dof indices
   * of the mother cell, i.e. the coarse level. while row indices belong to
   * the child cell, i.e. the fine level.
   */
  std::vector<std::shared_ptr<BlockSparseMatrix<double>>> prolongation_matrices;

  /**
   * Mapping for the <tt>copy_to/from_mg</tt>-functions. The indices into this
   * vector are (in this order): global block number, level number. The data
   * is first the global index inside the block, then the level index inside
   * the block.
   */
  std::vector<std::vector<std::vector<std::pair<unsigned int, unsigned int>>>>
    copy_indices;

  /**
   * The mg_constrained_dofs of the level systems.
   */

  SmartPointer<const MGConstrainedDoFs, MGTransferBlockBase>
    mg_constrained_dofs;
};

/**
 * Implementation of the MGTransferBase interface for block matrices and block
 * vectors.
 *
 * @warning This class is in an untested state. If you use it and you
 * encounter problems, please contact Guido Kanschat.
 *
 * In addition to the functionality of MGTransferPrebuilt, the operation may
 * be restricted to certain blocks of the vector.
 *
 * If the restricted mode is chosen, block vectors used in the transfer
 * routines may only have as many blocks as there are @p trues in the
 * selected-field.
 *
 * See MGTransferBase to find out which of the transfer classes is best for
 * your needs.
 *
 * @author Guido Kanschat, 2001, 2002
 */
template <typename number>
class MGTransferBlock : public MGTransferBase<BlockVector<number>>,
                        private MGTransferBlockBase
{
public:
  /**
   * Default constructor.
   */
  MGTransferBlock();

  /**
   * Destructor.
   */
  virtual ~MGTransferBlock() override;

  /**
   * Initialize additional #factors and #memory if the restriction of the
   * blocks is to be weighted differently.
   */
  void
  initialize(const std::vector<number> &   factors,
             VectorMemory<Vector<number>> &memory);

  /**
   * Build the prolongation matrices for each level.
   *
   * This function is a front-end for the same function in
   * MGTransferBlockBase.
   */
  template <int dim, int spacedim>
  void
  build(const DoFHandler<dim, spacedim> &dof_handler,
        const std::vector<bool> &        selected);

  /**
   * Build the prolongation matrices for each level.
   *
   * This function is a front-end for the same function in
   * MGTransferBlockBase.
   *
   * @deprecated Use the build() function instead.
   */
  template <int dim, int spacedim>
  DEAL_II_DEPRECATED void
  build_matrices(const DoFHandler<dim, spacedim> &dof,
                 const DoFHandler<dim, spacedim> &mg_dof,
                 const std::vector<bool> &        selected);

  virtual void
  prolongate(const unsigned int         to_level,
             BlockVector<number> &      dst,
             const BlockVector<number> &src) const override;

  virtual void
  restrict_and_add(const unsigned int         from_level,
                   BlockVector<number> &      dst,
                   const BlockVector<number> &src) const override;

  /**
   * Transfer from a vector on the global grid to a multilevel vector for the
   * active degrees of freedom. In particular, for a globally refined mesh only
   * the finest level in @p dst is filled  as a plain copy of @p src. All the
   * other level objects are left untouched.
   *
   * The action for discontinuous elements is as follows: on an active mesh
   * cell, the global vector entries are simply copied to the corresponding
   * entries of the level vector. Then, these values are restricted down to
   * the coarsest level.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_to_mg(const DoFHandler<dim, spacedim> &   dof_handler,
             MGLevelObject<BlockVector<number>> &dst,
             const BlockVector<number2> &        src) const;

  /**
   * Transfer from multi-level vector to normal vector.
   *
   * Copies data from active portions of a multilevel vector into the
   * respective positions of a global vector.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_from_mg(const DoFHandler<dim, spacedim> &         dof_handler,
               BlockVector<number2> &                    dst,
               const MGLevelObject<BlockVector<number>> &src) const;

  /**
   * Add a multi-level vector to a normal vector.
   *
   * Works as the previous function, but probably not for continuous elements.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_from_mg_add(const DoFHandler<dim, spacedim> &         dof_handler,
                   BlockVector<number2> &                    dst,
                   const MGLevelObject<BlockVector<number>> &src) const;

  using MGTransferBlockBase::memory_consumption;

private:
  /**
   * Optional multiplication factors for each block. Requires initialization
   * of #memory.
   */
  std::vector<number> factors;

  /**
   * Memory pool required if additional multiplication using #factors is
   * desired.
   */
  SmartPointer<VectorMemory<Vector<number>>, MGTransferBlock<number>> memory;
};


// TODO:[GK] Update documentation for copy_* functions

/**
 * Implementation of the MGTransferBase interface for block matrices and
 * simple vectors. This class uses MGTransferBlockBase selecting a single
 * block. The intergrid transfer operators are implemented for Vector objects,
 * The copy functions between regular and multigrid vectors for Vector and
 * BlockVector.
 *
 * See MGTransferBase to find out which of the transfer classes is best for
 * your needs.
 *
 * @author Guido Kanschat, 2001, 2002, 2003
 */
template <typename number>
class MGTransferBlockSelect : public MGTransferBase<Vector<number>>,
                              private MGTransferBlockBase
{
public:
  /**
   * Constructor without constraint matrices. Use this constructor only with
   * discontinuous finite elements or with no local refinement.
   */
  MGTransferBlockSelect();

  /**
   * Constructor with constraint matrices as well as mg_constrained_dofs.
   */
  MGTransferBlockSelect(const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Destructor.
   */
  virtual ~MGTransferBlockSelect() override = default;

  /**
   * Actually build the prolongation matrices for grouped blocks.
   *
   * This function is a front-end for the same function in
   * MGTransferBlockBase.
   *
   * @param dof_handler The DoFHandler to use.
   * @param selected Number of the block for which the transfer matrices
   * should be built.
   */
  template <int dim, int spacedim>
  void
  build(const DoFHandler<dim, spacedim> &dof_handler, unsigned int selected);

  /**
   * Actually build the prolongation matrices for grouped blocks.
   *
   * This function is a front-end for the same function in
   * MGTransferBlockBase.
   *
   * @param dof The DoFHandler for the active degrees of freedom to use.
   * @param mg_dof The DoFHandler for the level degrees of freedom to use.
   * @param selected Number of the block of the global vector to be copied from
   * and to the multilevel vector.
   *
   * @deprecated Use build() instead.
   */
  template <int dim, int spacedim>
  DEAL_II_DEPRECATED void
  build_matrices(const DoFHandler<dim, spacedim> &dof,
                 const DoFHandler<dim, spacedim> &mg_dof,
                 unsigned int                     selected);

  /**
   * Change selected block. Handle with care!
   */
  void
  select(const unsigned int block);

  virtual void
  prolongate(const unsigned int    to_level,
             Vector<number> &      dst,
             const Vector<number> &src) const override;

  virtual void
  restrict_and_add(const unsigned int    from_level,
                   Vector<number> &      dst,
                   const Vector<number> &src) const override;

  /**
   * Transfer a single block from a vector on the global grid to a multilevel
   * vector for the active degrees of freedom. In particular, for a globally
   * refined mesh only the finest level in @p dst is filled as a plain copy of
   * @p src. All the other level objects are left untouched.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_to_mg(const DoFHandler<dim, spacedim> &dof_handler,
             MGLevelObject<Vector<number>> &  dst,
             const Vector<number2> &          src) const;

  /**
   * Transfer from multilevel vector to normal vector.
   *
   * Copies data from active portions of an multilevel vector into the
   * respective positions of a Vector.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_from_mg(const DoFHandler<dim, spacedim> &    dof_handler,
               Vector<number2> &                    dst,
               const MGLevelObject<Vector<number>> &src) const;

  /**
   * Add a multi-level vector to a normal vector.
   *
   * Works as the previous function, but probably not for continuous elements.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_from_mg_add(const DoFHandler<dim, spacedim> &    dof_handler,
                   Vector<number2> &                    dst,
                   const MGLevelObject<Vector<number>> &src) const;

  /**
   * Transfer a block from a vector on the global grid to multilevel vectors.
   * Only the values for the active degrees of freedom of the block selected are
   * transferred. In particular, for a globally refined mesh only the finest
   * level in @p dst is filled as a plain copy of @p src. All the other level
   * objects are left untouched.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_to_mg(const DoFHandler<dim, spacedim> &dof_handler,
             MGLevelObject<Vector<number>> &  dst,
             const BlockVector<number2> &     src) const;

  /**
   * Transfer from multilevel vector to normal vector.
   *
   * Copies data from active portions of a multilevel vector into the
   * respective positions of a global BlockVector.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_from_mg(const DoFHandler<dim, spacedim> &    dof_handler,
               BlockVector<number2> &               dst,
               const MGLevelObject<Vector<number>> &src) const;

  /**
   * Add a multi-level vector to a normal vector.
   *
   * Works as the previous function, but probably not for continuous elements.
   */
  template <int dim, typename number2, int spacedim>
  void
  copy_from_mg_add(const DoFHandler<dim, spacedim> &    dof_handler,
                   BlockVector<number2> &               dst,
                   const MGLevelObject<Vector<number>> &src) const;

  /**
   * Memory used by this object.
   */
  std::size_t
  memory_consumption() const;

private:
  /**
   * Implementation of the public function.
   */
  template <int dim, class OutVector, int spacedim>
  void
  do_copy_from_mg(const DoFHandler<dim, spacedim> &    dof_handler,
                  OutVector &                          dst,
                  const MGLevelObject<Vector<number>> &src,
                  const unsigned int                   offset) const;

  /**
   * Implementation of the public function.
   */
  template <int dim, class OutVector, int spacedim>
  void
  do_copy_from_mg_add(const DoFHandler<dim, spacedim> &    dof_handler,
                      OutVector &                          dst,
                      const MGLevelObject<Vector<number>> &src,
                      const unsigned int                   offset) const;

  /**
   * Actual implementation of copy_to_mg().
   */
  template <int dim, class InVector, int spacedim>
  void
  do_copy_to_mg(const DoFHandler<dim, spacedim> &dof_handler,
                MGLevelObject<Vector<number>> &  dst,
                const InVector &                 src,
                const unsigned int               offset) const;
  /**
   * Selected block.
   */
  unsigned int selected_block;
};

/*@}*/

//------------------------- inline function definition ------------------------
template <typename number>
inline void
MGTransferBlockSelect<number>::select(const unsigned int block)
{
  selected_block = block;
}

DEAL_II_NAMESPACE_CLOSE

#endif
