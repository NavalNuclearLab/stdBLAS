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
#include "experimental/__p2630_bits/submdspan.hpp"
#include <experimental/linalg>
#include <iostream>
#include <vector>

namespace MdSpan = MDSPAN_IMPL_STANDARD_NAMESPACE;
namespace LinearAlgebra = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE :: linalg;

using MdSpan::mdspan;
using MdSpan::extents;
using MdSpan::full_extent;
using MdSpan::submdspan;
#if defined(__cpp_lib_span)
#include <span>
  using std::dynamic_extent;
#else
  using MdSpan::dynamic_extent;
#endif

int main(int argc, char* argv[]) {
  std::cout << "Matrix Vector Product MixedPrec" << std::endl;
  int M = 40;
  {
    // Create Data
    std::vector<float> A_vec(M * 8 * 4);
    std::vector<double> x_vec(M * 4);
    std::vector<double> y_vec(M * 8);

    // Create and initialize mdspan
    mdspan<float, extents<int, dynamic_extent, 8, 4>> A(A_vec.data(), M);
    mdspan<double, extents<int, 4, dynamic_extent>> x(x_vec.data(), M);
    mdspan<double, extents<int, dynamic_extent, 8>> y(y_vec.data(), M);
    for (int m = 0; m < A.extent(0); ++m) {
      for (int i = 0; i < A.extent(1); ++i) {
        for (int j = 0; j < A.extent(2); ++j) {
          A(m,i,j) = 1000.0 * m + 100.0 * i + j;
        }
      }
    }
    for (int i = 0; i < x.extent(0); ++i) {
      for (int m = 0; m < x.extent(1); ++m) {
        x(i,m) = 33.0 * i + 0.33 * m;
      }
    }
    for (int m = 0; m < y.extent(0); ++m) {
      for (int i = 0; i < y.extent(1); ++i) {
        y(m,i) = 33.0 * m + 0.33 * i;
      }
    }

    for (int m = 0; m < M; ++m) {
      auto A_m = submdspan(A, m, full_extent, full_extent);
      auto x_m = submdspan(x, full_extent, m);
      auto y_m = submdspan(y, m, full_extent);
      // y_m = A * x_m
      LinearAlgebra::matrix_vector_product(A_m, x_m, y_m);
    }

    for (int i = 0; i < y.extent(0); i += 5) {
      std::cout << i << " " << y(i,1) << std::endl;
    }
  }
}
