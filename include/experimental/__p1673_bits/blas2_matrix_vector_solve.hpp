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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_

#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X,
         class BinaryDivideOp>
void trsv_upper_triangular_left_side(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> X,
  BinaryDivideOp divide)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type A_num_rows = A.extent(0);
  const size_type A_num_cols = A.extent(1);

  // One advantage of using signed index types is that you can write
  // descending loops with zero-based indices.
  // (AMK 6/8/21) i can't be a nonnegative type because the loop would be infinite
  for (ptrdiff_t i = A_num_rows - 1; i >= 0; --i) {
    // TODO this would be a great opportunity for an implementer to
    // add value, by accumulating in extended precision (or at least
    // in a type with the max precision of X and B).
    using sum_type = decltype (B(i) - A(0,0) * X(0));
    //using sum_type = typename out_object_t::element_type;
    sum_type t (B(i));
    for (size_type j = i + 1; j < A_num_cols; ++j) {
      t = t - A(i,j) * X(j);
    }
    if constexpr (explicit_diagonal) {
      X(i) = divide(t, A(i,i));
    }
    else {
      X(i) = t;
    }
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X>
void trsv_upper_triangular_left_side(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> X)
{
  auto divide = [](const auto& x, const auto& y) { return x / y; };
  trsv_upper_triangular_left_side(A, d, B, X, divide);
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X,
         class BinaryDivideOp>
void trsv_lower_triangular_left_side(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> X,
  BinaryDivideOp divide)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type A_num_rows = A.extent(0);
  const size_type A_num_cols = A.extent(1);

  for (size_type i = 0; i < A_num_rows; ++i) {
    // TODO this would be a great opportunity for an implementer to
    // add value, by accumulating in extended precision (or at least
    // in a type with the max precision of X and B).
    using sum_type = decltype (B(i) - A(0,0) * X(0));
    //using sum_type = typename out_object_t::element_type;
    sum_type t (B(i));
    for (size_type j = 0; j < i; ++j) {
      t = t - A(i,j) * X(j);
    }
    if constexpr (explicit_diagonal) {
      X(i) = divide(t, A(i,i));
    }
    else {
      X(i) = t;
    }
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X>
void trsv_lower_triangular_left_side(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> X)
{
  auto divide = [](const auto& x, const auto& y) { return x / y; };
  trsv_lower_triangular_left_side(A, d, B, X, divide);
}

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t, class = void>
struct is_custom_tri_mat_vec_solve_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t>
struct is_custom_tri_mat_vec_solve_avail<
  Exec, A_t, Tri_t, D_t, B_t, X_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(triangular_matrix_vector_solve
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Tri_t>(),
		std::declval<D_t>(),
		std::declval<B_t>(),
		std::declval<X_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

} // end anonymous namespace

// Special case: ExecutionPolicy = inline_exec_t

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X,
         class BinaryDivideOp>
void triangular_matrix_vector_solve(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x,
  BinaryDivideOp divide)
{
  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    trsv_lower_triangular_left_side(A, d, b, x, divide);
  }
  else {
    trsv_upper_triangular_left_side(A, d, b, x, divide);
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X>
void triangular_matrix_vector_solve(
  impl::inline_exec_t&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x)
{
  auto divide = [](const auto& x, const auto& y) { return x / y; };
  triangular_matrix_vector_solve(std::forward<impl::inline_exec_t>(exec), A, t, d, b, x, divide);
}

// Overloads taking an ExecutionPolicy

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X,
         class BinaryDivideOp>
void triangular_matrix_vector_solve(
  ExecutionPolicy&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x,
  BinaryDivideOp divide)
{
  // FIXME (mfh 2022/06/13) We don't yet have a parallel version
  // that takes a generic divide operator.
  triangular_matrix_vector_solve(impl::inline_exec_t{}, A, t, d, b, x, divide);
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X>
void triangular_matrix_vector_solve(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x)
{
  constexpr bool use_custom = is_custom_tri_mat_vec_solve_avail<
    decltype(impl::map_execpolicy_with_check(exec)),
    decltype(A), decltype(t), decltype(d), decltype(b), decltype(x)
    >::value;

  if constexpr (use_custom) {
    triangular_matrix_vector_solve(impl::map_execpolicy_with_check(exec), A, t, d, b, x);
  }
  else {
    triangular_matrix_vector_solve(impl::inline_exec_t{},
				   A, t, d, b, x);
  }
}

// Overloads not taking an ExecutionPolicy

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X,
         class BinaryDivideOp>
void triangular_matrix_vector_solve(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x,
  BinaryDivideOp divide)
{
  triangular_matrix_vector_solve(impl::default_exec_t{},
				 A, t, d, b, x, divide);
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_B,
         class SizeType_B, ::std::size_t ext_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         class SizeType_X, ::std::size_t ext_X,
         class Layout_X,
         class Accessor_X>
void triangular_matrix_vector_solve(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_B, extents<SizeType_B, ext_B>, Layout_B, Accessor_B> b,
  mdspan<ElementType_X, extents<SizeType_X, ext_X>, Layout_X, Accessor_X> x)
{
  triangular_matrix_vector_solve(impl::default_exec_t{},
				 A, t, d, b, x);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_
