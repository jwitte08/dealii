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



// check the correctness of the generic d-mode product kernel
// contract_general_impl() for anisotropic tensors

#include <deal.II/base/vectorization.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <iostream>
#include <sstream>

#include "../tests.h"


template <int  mode,
          int  M,
          int  N0,
          int  N1,
          int  N2,
          bool contract_over_rows,
          bool add>
void
test()
{
  const auto random_macro_value = []() {
    VectorizedArray<double> val;
    for (unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
      val[v] = random_value<double>();
    return val;
  };

  /// for debugging
  const auto tensor_to_string = [](const auto &tensor) {
    const unsigned int lane = 0;
    std::ostringstream oss;
    oss << std::endl;
    for (unsigned int layer = 0; layer < tensor.size(2); ++layer)
      {
        FullMatrix<double> matrix(tensor.size(0), tensor.size(1));
        for (unsigned int i = 0; i < matrix.m(); ++i)
          for (unsigned int j = 0; j < matrix.n(); ++j)
            matrix(i, j) = tensor(i, j, layer)[lane];
        oss << "layer = " << layer << std::endl;
        matrix.print_formatted(oss);
      }
    return oss.str();
  };

  constexpr int N  = (mode == 0) ? N0 : ((mode == 1) ? N1 : N2);
  constexpr int MM = contract_over_rows ? N : M;
  constexpr int NN = contract_over_rows ? M : N;
  deallog << "Test " << M << " x " << N << std::endl;

  /// fill "shape data"
  AlignedVector<double> shape(M * N);
  for (unsigned int i = 0; i < M; ++i)
    for (unsigned int j = 0; j < N; ++j)
      shape[i * N + j] =
        /*i == j ? 1. : 0.;*/ -1. + 2. * random_value<double>();
  Table<2, double> shape_ref(MM, NN);
  for (unsigned int i = 0; i < M; ++i)
    for (unsigned int j = 0; j < N; ++j)
      {
        if (contract_over_rows)
          shape_ref(j, i) = shape[i * N + j];
        else
          shape_ref(i, j) = shape[i * N + j];
      }
  // {
  //   std::ostringstream oss;
  //   FullMatrix<double> matrix(MM, NN);
  //   for (unsigned int i = 0; i < MM; ++i)
  //     for (unsigned int j = 0; j < NN; ++j)
  // 	matrix(i,j) = shape_ref(i,j);
  //   matrix.print_formatted(oss);
  //   deallog << MM << " x " << NN << " shape_ref:\n" << oss.str();
  // }

  /// fill input tensor
  constexpr int                          NN0 = (mode == 0) ? NN : N0;
  constexpr int                          NN1 = (mode == 1) ? NN : N1;
  constexpr int                          NN2 = (mode == 2) ? NN : N2;
  AlignedVector<VectorizedArray<double>> in(NN0 * NN1 * NN2);
  for (unsigned int k = 0; k < NN2; ++k)
    for (unsigned int j = 0; j < NN1; ++j)
      for (unsigned int i = 0; i < NN0; ++i)
        in[k * NN1 * NN0 + j * NN0 + i] =
          random_macro_value(); // 1. * i + 10. * j + 100. * k;
  Table<3, VectorizedArray<double>> in_ref(NN0, NN1, NN2);
  for (unsigned int k = 0; k < NN2; ++k)
    for (unsigned int j = 0; j < NN1; ++j)
      for (unsigned int i = 0; i < NN0; ++i)
        in_ref(i, j, k) = in[k * NN1 * NN0 + j * NN0 + i];
  // deallog << "in_ref:" << tensor_to_string(in_ref);

  /// fill output tensor
  constexpr int                          MM0 = (mode == 0) ? MM : N0;
  constexpr int                          MM1 = (mode == 1) ? MM : N1;
  constexpr int                          MM2 = (mode == 2) ? MM : N2;
  AlignedVector<VectorizedArray<double>> out(MM0 * MM1 * MM2);
  for (unsigned int k = 0; k < MM2; ++k)
    for (unsigned int j = 0; j < MM1; ++j)
      for (unsigned int i = 0; i < MM0; ++i)
        out[k * MM1 * MM0 + j * MM0 + i] =
          random_macro_value(); // 1. * i + 10. * j + 100. * k;
  Table<3, VectorizedArray<double>> out_ref(MM0, MM1, MM2);
  for (unsigned int k = 0; k < MM2; ++k)
    for (unsigned int j = 0; j < MM1; ++j)
      for (unsigned int i = 0; i < MM0; ++i)
        out_ref(i, j, k) = out[k * MM1 * MM0 + j * MM0 + i];

  /// compute reference contraction
  if (mode == 0)
    for (unsigned int k = 0; k < N2; ++k)
      for (unsigned int j = 0; j < N1; ++j)
        for (unsigned int i = 0; i < MM; ++i)
          {
            auto sum = make_vectorized_array<double>(0.);
            for (unsigned int nn = 0; nn < NN; ++nn)
              sum += shape_ref(i, nn) * in_ref(nn, j, k);
            if (add)
              out_ref(i, j, k) += sum;
            else
              out_ref(i, j, k) = sum;
          }
  else if (mode == 1)
    for (unsigned int k = 0; k < N2; ++k)
      for (unsigned int j = 0; j < MM; ++j)
        for (unsigned int i = 0; i < N0; ++i)
          {
            auto sum = make_vectorized_array<double>(0.);
            for (unsigned int nn = 0; nn < NN; ++nn)
              sum += shape_ref(j, nn) * in_ref(i, nn, k);
            if (add)
              out_ref(i, j, k) += sum;
            else
              out_ref(i, j, k) = sum;
          }
  else if (mode == 2)
    for (unsigned int k = 0; k < MM; ++k)
      for (unsigned int j = 0; j < N1; ++j)
        for (unsigned int i = 0; i < N0; ++i)
          {
            auto sum = make_vectorized_array<double>(0.);
            for (unsigned int nn = 0; nn < NN; ++nn)
              sum += shape_ref(k, nn) * in_ref(i, j, nn);
            if (add)
              out_ref(i, j, k) += sum;
            else
              out_ref(i, j, k) = sum;
          }
  else
    AssertThrow(false, ExcMessage("Not implemented."));
  // deallog << tensor_to_string(out_ref) << std::endl;

  /// compute tensor contraction
  constexpr int mode_size_pre =
    internal::collapse_sizes_pre(mode, NN0, NN1, NN2);
  constexpr int mode_size_post =
    internal::collapse_sizes_post(mode, NN0, NN1, NN2);
  internal::contract_general_impl<3,
                                  VectorizedArray<double>,
                                  M,
                                  N,
                                  mode,
                                  mode_size_pre,
                                  mode_size_post,
                                  contract_over_rows,
                                  add,
                                  double>(shape.data(), in.data(), out.data());
  // for (auto val = out.begin(); val != out.end(); ++val)
  //   deallog << (*val)[0] << std::endl;

  /// compare results element-wise
  deallog << "Errors: ";
  for (unsigned int k = 0; k < MM2; ++k)
    for (unsigned int j = 0; j < MM1; ++j)
      for (unsigned int i = 0; i < MM0; ++i)
        {
          deallog << out_ref(i, j, k)[0] - out[k * MM1 * MM0 + j * MM0 + i][0]
                  << " ";
          for (unsigned int v = 0;
               v < VectorizedArray<double>::n_array_elements;
               ++v)
            AssertThrow(std::abs(out_ref(i, j, k)[v] -
                                 out[k * MM1 * MM0 + j * MM0 + i][v]) < 1e-12,
                        ExcInternalError());
        }
  deallog << std::endl;
}



int
main()
{
  initlog();

  // test<mode, M, N0, N1, N2, contract_over_rows, add>

  deallog.push("over_cols");
  // mode-0 product & square matrix
  {
    constexpr int M = 1;
    test<0, M, M, M, M, false, false>();
    test<0, M, M, 5, 3, false, false>();
    test<0, M, M, 2, 7, false, false>();
  }
  {
    constexpr int M = 4;
    test<0, M, M, M, M, false, false>();
    test<0, M, M, 5, 3, false, false>();
    test<0, M, M, 2, 7, false, false>();
  }

  // mode-1 product & square matrix
  {
    constexpr int M = 1;
    test<1, M, M, M, M, false, false>();
    test<1, M, 5, M, 3, false, false>();
    test<1, M, 2, M, 7, false, false>();
  }
  {
    constexpr int M = 4;
    test<1, M, M, M, M, false, false>();
    test<1, M, 5, M, 3, false, false>();
    test<1, M, 2, M, 7, false, false>();
  }

  // mode-2 product & square matrix
  {
    constexpr int M = 1;
    test<2, M, M, M, M, false, false>();
    test<2, M, 5, 3, M, false, false>();
    test<2, M, 2, 7, M, false, false>();
  }
  {
    constexpr int M = 4;
    test<2, M, M, M, M, false, false>();
    test<2, M, 5, 3, M, false, false>();
    test<2, M, 2, 7, M, false, false>();
  }

  // all modes & arbitrary matrix
  {
    test<0, 4, 3, 2, 1, false, true>();
    test<0, 1, 2, 3, 4, false, true>();
    test<0, 3, 5, 2, 7, false, true>();
    test<0, 2, 5, 7, 3, false, true>();
    test<1, 4, 3, 2, 1, false, true>();
    test<1, 1, 2, 3, 4, false, true>();
    test<1, 3, 5, 2, 7, false, true>();
    test<1, 2, 5, 7, 3, false, true>();
    test<2, 4, 3, 2, 1, false, true>();
    test<2, 1, 2, 3, 4, false, true>();
    test<2, 3, 5, 2, 7, false, true>();
    test<2, 2, 5, 7, 3, false, true>();
  }
  deallog.pop();


  deallog.push("over_rows");
  // mode-0 product & square matrix
  {
    constexpr int M = 1;
    test<0, M, M, M, M, true, true>();
    test<0, M, M, 5, 3, true, true>();
    test<0, M, M, 2, 7, true, true>();
  }
  {
    constexpr int M = 4;
    test<0, M, M, M, M, true, true>();
    test<0, M, M, 5, 3, true, true>();
    test<0, M, M, 2, 7, true, true>();
  }

  // mode-1 product & square matrix
  {
    constexpr int M = 1;
    test<1, M, M, M, M, true, true>();
    test<1, M, 5, M, 3, true, true>();
    test<1, M, 2, M, 7, true, true>();
  }
  {
    constexpr int M = 4;
    test<1, M, M, M, M, true, true>();
    test<1, M, 5, M, 3, true, true>();
    test<1, M, 2, M, 7, true, true>();
  }

  // mode-2 product & square matrix
  {
    constexpr int M = 1;
    test<2, M, M, M, M, true, true>();
    test<2, M, 5, 3, M, true, true>();
    test<2, M, 2, 7, M, true, true>();
  }
  {
    constexpr int M = 4;
    test<2, M, M, M, M, true, true>();
    test<2, M, 5, 3, M, true, true>();
    test<2, M, 2, 7, M, true, true>();
  }

  // all modes & arbitrary matrix
  {
    test<0, 4, 3, 2, 1, true, false>();
    test<0, 1, 2, 3, 4, true, false>();
    test<0, 3, 5, 2, 7, true, false>();
    test<0, 2, 5, 7, 3, true, false>();
    test<1, 4, 3, 2, 1, true, false>();
    test<1, 1, 2, 3, 4, true, false>();
    test<1, 3, 5, 2, 7, true, false>();
    test<1, 2, 5, 7, 3, true, false>();
    test<2, 4, 3, 2, 1, true, false>();
    test<2, 1, 2, 3, 4, true, false>();
    test<2, 3, 5, 2, 7, true, false>();
    test<2, 2, 5, 7, 3, true, false>();
  }
  deallog.pop();

  return 0;
}
