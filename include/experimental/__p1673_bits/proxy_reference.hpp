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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_PROXY_REFERENCE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_PROXY_REFERENCE_HPP_

#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
#  include <atomic>
#endif
#if __cplusplus >= 202002L
#  include <concepts>
#endif // __cplusplus >= 202002L
#include <cstdint>
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl {

template<class T>
static constexpr bool is_atomic_ref_not_arithmetic_v = false;

#if defined(__cpp_lib_atomic_ref) && defined(LINALG_ENABLE_ATOMIC_REF)
template<class U>
static constexpr bool is_atomic_ref_not_arithmetic_v<std::atomic_ref<U>> = ! std::is_arithmetic_v<U>;
#endif

// A "tag" for identifying the proxy reference types in this proposal.
// It's helpful for this tag to be a complete type, so that we can use
// it inside proxy_reference (proxy_reference isn't really complete
// inside itself).
class proxy_reference_base {};

// Mixin that will provide all the arithmetic operators
// for the proxy reference types, to be defined below.
//
// NOTE (mfh 2022/06/03) Consider getting rid of Value, since it can
// be deduced as the return type of Derived::to_value(Reference).
// However, Derived isn't really a complete type in this class,
// so doing this isn't so easy.
template<class Reference, class Value, class Derived>
class proxy_reference : proxy_reference_base {
private:
  static_assert(std::is_same_v<Value, std::remove_cv_t<Value>>);
  using this_type = proxy_reference<Reference, Value, Derived>;
  
  Reference reference_;

public:
  using reference_type = Reference;
  using value_type = Value;
  using derived_type = Derived;

  // NOTE (mfh 2022/06/03) "explicit" may prevent implicit conversions
  // that cause ambiguity among overloaded operator selection.
  explicit proxy_reference(Reference reference) : reference_(reference) {}

  operator value_type() const {
    return static_cast<const Derived&>(*this).to_value(reference_);
  }

  ////////////////////////////////////////////////////////////
  // Unary negation
  ////////////////////////////////////////////////////////////
  
  friend auto operator-(const derived_type& cs)
  {
    return -value_type(cs);
  }

  // Case 1: rhs is a subclass of proxy_reference of a possibly different type.    
#define P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR_CASE1( SYMBOL ) \
  template<class Rhs, std::enable_if_t<std::is_base_of_v<proxy_reference_base, Rhs>, bool> = true> \
  friend auto \
  operator SYMBOL (derived_type lhs, Rhs rhs) \
  { \
    using rhs_value_type = typename Rhs::value_type; \
    return value_type(lhs) SYMBOL rhs_value_type(rhs); \
  }
  
  // Case 2: rhs is NOT a subclass of proxy_reference
  //
  // Another way to work around the lack of overloaded operators for
  // atomic_ref<complex<R>> would be to provide a function that makes
  // an mdspan "atomic," and for that function to use something other
  // than atomic_ref if the value_type is complex<R>.
#define P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR_CASE2( SYMBOL ) \
  template<class Rhs, std::enable_if_t<! std::is_base_of_v<proxy_reference_base, Rhs>, bool> = true> \
  friend auto				   \
  operator SYMBOL (derived_type lhs, Rhs rhs) \
  { \
    if constexpr (impl::is_atomic_ref_not_arithmetic_v<Rhs>) { \
      return value_type(lhs) SYMBOL rhs.load(); \
    } else { \
      return value_type(lhs) SYMBOL rhs; \
    } \
  }

  // Case 3: lhs is not a subclass of proxy_reference, rhs is derived_type.
#define P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR_CASE3( SYMBOL ) \
  template<class Lhs, std::enable_if_t<! std::is_base_of_v<proxy_reference_base, Lhs>, bool> = true> \
  friend auto				   \
  operator SYMBOL (Lhs lhs, derived_type rhs) \
  { \
    if constexpr (impl::is_atomic_ref_not_arithmetic_v<Lhs>) { \
      return lhs.load() SYMBOL value_type(rhs); \
    } else { \
      return lhs SYMBOL value_type(rhs); \
    } \
  }

#define P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR( SYMBOL ) \
  P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR_CASE1( SYMBOL ) \
  P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR_CASE2( SYMBOL ) \
  P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR_CASE3( SYMBOL )
  
  ////////////////////////////////////////////////////////////
  // Binary plus, minus, times, and divide
  ////////////////////////////////////////////////////////////

  P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR( + )
  P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR( - )
  P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR( * )
  P1673_PROXY_REFERENCE_ARITHMETIC_OPERATOR( / )

  friend auto abs(const derived_type& x) {
    if constexpr (std::is_unsigned_v<value_type>) {
      return value_type(static_cast<const this_type&>(x));
    } else {
      return abs(value_type(static_cast<const this_type&>(x)));
    }
  }

  friend auto real(const derived_type& x) {
    return impl::real_if_needed(value_type(static_cast<const this_type&>(x)));
  }
  
  friend auto imag(const derived_type& x) {
    return impl::imag_if_needed(value_type(static_cast<const this_type&>(x)));
  }

  friend auto conj(const derived_type& x) {
    return impl::conj_if_needed(value_type(static_cast<const this_type&>(x)));
  }
};

} // namespace impl

// Proxy reference type representing the conjugate (in the sense of
// complex arithmetic) of a value.
//
// The point of ReferenceValue is so that we can cast the input of
// to_value to a value immediately, before we apply any
// transformations.  This has two goals.
//
// 1. Ensure the original order of operations (as if computing nonlazily)
//
// 2. Make it possible to use reference types that don't have
//    arithmetic operators defined, such as
//    std::atomic_ref<std::complex<R>>.  (atomic_ref<T> for arithmetic
//    types T _does_ have arithmetic operators.)
template<class Reference, class ReferenceValue>
class conjugated_scalar :
  public impl::proxy_reference<Reference, ReferenceValue, conjugated_scalar<Reference, ReferenceValue>>
{
private:
  using my_type = conjugated_scalar<Reference, ReferenceValue>;
  using base_type = impl::proxy_reference<Reference, ReferenceValue, my_type>;

public:
  explicit conjugated_scalar(Reference reference) : base_type(reference) {}

  // NOTE (mfh 2022/06/03) Consider moving this to proxy_reference,
  // since it's duplicated in all the proxy reference "base" types.
  // Doing so isn't easy, because this class is an incomplete type
  // inside proxy_reference at the time when we need it to deduce this
  // type.
  using value_type = decltype(impl::conj_if_needed(ReferenceValue(std::declval<Reference>())));
  static auto to_value (Reference reference) {
    return impl::conj_if_needed(ReferenceValue(reference));
  }
};

// Proxy reference type representing the product of a scaling factor
// and a value.
template<class ScalingFactor, class Reference, class ReferenceValue>
class scaled_scalar :
  public impl::proxy_reference<Reference, ReferenceValue,
    scaled_scalar<ScalingFactor, Reference, ReferenceValue>>
{
private:
  ScalingFactor scaling_factor_;

  using my_type = scaled_scalar<ScalingFactor, Reference, ReferenceValue>;
  using base_type = impl::proxy_reference<Reference, ReferenceValue, my_type>;

public:
  explicit scaled_scalar(ScalingFactor scaling_factor, Reference reference) :
    base_type(reference),
    scaling_factor_(std::move(scaling_factor))
  {}

  using value_type = decltype(scaling_factor_ * ReferenceValue(std::declval<Reference>()));
  value_type to_value (Reference reference) const {
    return scaling_factor_ * ReferenceValue(reference);
  }

  // scaled_scalar operator== is just for tests.
  friend bool operator==(
    const my_type& lhs,
    const value_type& rhs)
  {
    return value_type(static_cast<const base_type&>(lhs)) == rhs;
  }
};

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_PROXY_REFERENCE_HPP_
