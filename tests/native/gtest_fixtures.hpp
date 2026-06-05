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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_FIXTURES_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_FIXTURES_HPP_

#include "gtest/gtest.h"

// Tests currently use parentheses (e.g., A(i,j))
// for the array access operator,
// instead of square brackets (e.g., A[i,j]).
// This must be defined before including any mdspan headers.
#define MDSPAN_USE_PAREN_OPERATOR 1

#include <mdspan/mdspan.hpp>
#include "experimental/__p2630_bits/submdspan.hpp"
#include <experimental/linalg>
#include <array>
#include <complex>
#include <vector>

namespace KokkosStd = MDSPAN_IMPL_STANDARD_NAMESPACE;
namespace KokkosEx = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE;

namespace MdSpan = MDSPAN_IMPL_STANDARD_NAMESPACE;
namespace MdSpanEx = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE;
namespace LinearAlgebra = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE :: linalg;

using MdSpan::default_accessor;
using MdSpan::dextents; // not in experimental namespace
#if defined(__cpp_lib_span)
#include <span>
using std::dynamic_extent;
#else
using MdSpan::dynamic_extent;
#endif
using MdSpan::extents;
using MdSpan::full_extent; // not in experimental namespace
using MdSpan::layout_left;
using MdSpan::layout_right;
using MdSpan::layout_stride;
using MdSpan::mdspan;
using MdSpan::submdspan;

using MdSpan::layout_left_padded; // not in experimental namespace
using MdSpan::layout_right_padded;

using LinearAlgebra::upper_triangle_t;
using LinearAlgebra::lower_triangle_t;
using LinearAlgebra::implicit_unit_diagonal_t;
using LinearAlgebra::explicit_diagonal_t;

using dbl_vector_t = mdspan<double, extents<std::size_t, dynamic_extent>>;
using cpx_vector_t = mdspan<std::complex<double>, extents<std::size_t, dynamic_extent>>;
constexpr ptrdiff_t NROWS = 10u;

template<class ElementType,
         class Extents,
         class Layout,
         class Accessor>
std::string to_string( mdspan<ElementType, Extents, Layout, Accessor> x) {
  std::stringstream ss;
  constexpr auto ndims = x.rank();
  if constexpr(ndims == 1) {
    for(std::size_t i = 0; i < x.extent(0); ++i) {
      ss << x(i) << " ";
    }
    ss << "\n";
  }
  else if constexpr(ndims == 2) {
    for(std::size_t i = 0; i < x.extent(0); ++i) {
      for(std::size_t j = 0; j < x.extent(1); ++j) {
        ss << x(i, j) << " ";
      }
      ss << "\n";
    }
  }
  else {
    printf("to_string(mdspan): only implemented for rank 1 and 2\n");
  }
  return ss.str();
}

template<class ElementType,
         class Extents,
         class Layout,
         class Accessor,
         class Triangle>
std::string to_string( mdspan<ElementType, Extents, Layout, Accessor> x, 
                       Triangle /* t */) {
  std::stringstream ss;
  constexpr auto ndims = x.rank();
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  if constexpr(ndims == 2) {
    for(std::size_t i = 0; i < x.extent(0); ++i) {
      for(std::size_t j = 0; j < x.extent(1); ++j) {
        if constexpr(lower_tri) {
          if (j > i) {
            ss << x(j, i) << " ";
            continue;
          }
        }
        else {
          if (i > j) {
            ss << x(j, i) << " ";
            continue;
          }
        }
        ss << x(i, j) << " ";
      }
      ss << "\n";
    }
  }
  else {
    printf("to_string(mdspan, Triangle): only implemented for rank 2\n");
  }
  return ss.str();
}

template<class ElementType,
         class Extents,
         class Layout,
         class Accessor,
         class Triangle,
         class Diagonal>
std::string to_string( mdspan<ElementType, Extents, Layout, Accessor> x, 
                       Triangle /* t */, Diagonal /* d */) {
  std::stringstream ss;
  constexpr auto ndims = x.rank();
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  constexpr bool implicit_unit_diag =
    std::is_same_v<Diagonal, implicit_unit_diagonal_t>;
  if constexpr(ndims == 2) {
    for(std::size_t i = 0; i < x.extent(0); ++i) {
      for(std::size_t j = 0; j < x.extent(1); ++j) {
        if constexpr(implicit_unit_diag) {
          if (i == j) {
            ss << "1 ";
            continue;
          }
        }
        if constexpr(lower_tri) {
          if (j > i) {
            ss << "0 ";
            continue;
          }
        }
        else {
          if (i > j) {
            ss << "0 ";
            continue;
          }
        }
        ss << x(i, j) << " ";
      }
      ss << "\n";
    }
  }
  else {
    printf("to_string(mdspan, Triangle, Diagonal): only implemented for rank 2\n");
  }
  return ss.str();
}

// 1-norm:   4.6
// inf-norm: 0.9
class unsigned_double_vector : public ::testing::Test {
protected:
  unsigned_double_vector() :
    storage(10),
    v(storage.data(), 10)
  {
    v(0) = 0.5;
    v(1) = 0.2;
    v(2) = 0.1;
    v(3) = 0.4;
    v(4) = 0.8;
    v(5) = 0.7;
    v(6) = 0.3;
    v(7) = 0.5;
    v(8) = 0.2;
    v(9) = 0.9;
  }

  std::vector<double> storage;
  dbl_vector_t v;
}; // end class unsigned_double_vector

// 1-norm:   4.6
// inf-norm: 0.9
class signed_double_vector : public ::testing::Test {
protected:
  signed_double_vector() :
    storage(10),
    v(storage.data(), 10)
  {
    v(0) =  0.5;
    v(1) =  0.2;
    v(2) =  0.1;
    v(3) =  0.4;
    v(4) = -0.8;
    v(5) = -0.7;
    v(6) = -0.3;
    v(7) =  0.5;
    v(8) =  0.2;
    v(9) = -0.9;
  }

  std::vector<double> storage;
  dbl_vector_t v;
}; // end class signed_double_vector

// 1-norm:   3.5188912597625004
// 2-norm:   1.6673332000533068
// inf-norm: 1.063014581273465
class signed_complex_vector : public ::testing::Test {
protected:
  signed_complex_vector() :
    storage(6),
    v(storage.data(), 6)
  {
    v(0) = std::complex<double>( 0.5,  0.2);
    v(1) = std::complex<double>( 0.1,  0.4);
    v(2) = std::complex<double>(-0.8,  0.7);
    v(3) = std::complex<double>(-0.79, -0.711);
    v(4) = std::complex<double>(-0.3,  0.5);
    v(5) = std::complex<double>( 0.2, -0.9);
  }

  std::vector<std::complex<double>> storage;
  cpx_vector_t v;
}; // end class signed_double_vector

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_FIXTURES_HPP_
