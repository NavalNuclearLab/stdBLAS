//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ************************************************************************
//@HEADER

// Examples currently use parentheses (e.g., A(i,j))
// for the array access operator,
// instead of square brackets (e.g., A[i,j]).
// This must be defined before including any mdspan headers.
#define MDSPAN_USE_PAREN_OPERATOR 1

#include <mdspan/mdspan.hpp>
#include <experimental/linalg>
#include <iostream>
#include <vector>

#ifdef LINALG_HAS_EXECUTION
#  include <execution>
#endif

namespace MdSpan = MDSPAN_IMPL_STANDARD_NAMESPACE;
namespace LinearAlgebra = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE :: linalg;

using MdSpan::mdspan;
using MdSpan::extents;
#if defined(__cpp_lib_span)
#include <span>
  using std::dynamic_extent;
#else
  using MdSpan::dynamic_extent;
#endif

int main(int argc, char* argv[]) {
  std::cout << "Matrix Vector Product Basic" << std::endl;
  int N = 40, M = 20;
  {
    // Create Data
    std::vector<double> A_vec(N*M);
    std::vector<double> x_vec(M);
    std::vector<double> y_vec(N);

    // Create and initialize mdspan
    // Would look simple with CTAD, GCC 11.1 works but some others are buggy
    mdspan<double, extents<int, dynamic_extent,dynamic_extent>> A(A_vec.data(), N, M);
    mdspan<double, extents<int, dynamic_extent>> x(x_vec.data(), M);
    mdspan<double, extents<int, dynamic_extent>> y(y_vec.data(), N);
    for (int i = 0; i < A.extent(0); ++i) {
      for (int j = 0; j < A.extent(1); ++j) {
        A(i,j) = 100.0 * i + j;
      }
    }
    for (int i = 0; i < x.extent(0); ++i) {
      x(i) = 1.0 * i;
    }
    for (int i = 0; i < y.extent(0); ++i) {
      y(i) = -1.0 * i;
    }

    // y = A * x
    LinearAlgebra::matrix_vector_product(A, x, y);

    // y = 0.5 * y + 2 * A * x
#ifdef LINALG_HAS_EXECUTION
    LinearAlgebra::matrix_vector_product(std::execution::par,
      LinearAlgebra::scaled(2.0, A), x,
      LinearAlgebra::scaled(0.5, y), y);
#else
    LinearAlgebra::matrix_vector_product(
      LinearAlgebra::scaled(2.0, A), x,
      LinearAlgebra::scaled(0.5, y), y);
#endif
    for (int i = 0; i < y.extent(0); i += 5) {
      std::cout << i << " " << y(i) << std::endl;
    }
  }
}
