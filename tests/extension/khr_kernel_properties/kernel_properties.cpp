/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Compile-time tests for the sycl_khr_kernel_properties extension: the
//  feature test macro, the khr::property::tag::kernel tag type, and the
//  khr::property::work_group_size, khr::property::work_group_size_hint and
//  khr::property::sub_group_size properties and their keys. The effects of
//  embedding the properties in a kernel are tested in runtime.cpp.
//
//  The extension requires its APIs to be provided by <sycl/sycl.hpp> and, if
//  the implementation also implements sycl_khr_split_headers, by
//  <sycl/khr/kernel_properties.hpp>. A test can only show that a split header
//  provides an API if <sycl/sycl.hpp> is not included in the same translation
//  unit, so this file includes only the split headers when they are available
//  and <sycl/sycl.hpp> otherwise. The khr::properties class and the property
//  traits are provided by <sycl/khr/properties.hpp> (see sycl_khr_properties),
//  sycl::queue by <sycl/khr/split_headers/queue.hpp>, and
//  khr::property::tag::reduction by <sycl/khr/split_headers/reduction.hpp>.
//
*******************************************************************************/

#if __has_include(<sycl/khr/split_headers/version.hpp>)
#include <sycl/khr/split_headers/version.hpp>
#endif

#ifdef SYCL_KHR_SPLIT_HEADERS
#include <sycl/khr/kernel_properties.hpp>
#include <sycl/khr/properties.hpp>
#include <sycl/khr/split_headers/queue.hpp>
#include <sycl/khr/split_headers/reduction.hpp>
#else
#include <sycl/sycl.hpp>
#endif

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef SYCL_KHR_KERNEL_PROPERTIES
#error "SYCL_KHR_KERNEL_PROPERTIES is not defined"
#endif
static_assert(SYCL_KHR_KERNEL_PROPERTIES == 1,
              "SYCL_KHR_KERNEL_PROPERTIES must be equal to 1");

namespace khr_kernel_properties::tests {

namespace khr = sycl::khr;
namespace prop = sycl::khr::property;
namespace key = sycl::khr::property::key;
using kernel_tag = sycl::khr::property::tag::kernel;

template <std::size_t... Dims>
using wg_size_t = std::remove_cv_t<decltype(prop::work_group_size<Dims...>)>;
template <std::size_t... Dims>
using wg_size_hint_t =
    std::remove_cv_t<decltype(prop::work_group_size_hint<Dims...>)>;
template <std::uint32_t Size>
using sg_size_t = std::remove_cv_t<decltype(prop::sub_group_size<Size>)>;

/** A type that is neither a property nor a properties list. */
struct not_a_property {};

template <template <typename> class Trait, typename T, bool Expected>
void check_unary_trait() {
  using base_t = std::integral_constant<bool, Expected>;
  STATIC_CHECK(std::is_base_of_v<base_t, Trait<T>>);
  STATIC_CHECK(Trait<T>::value == Expected);
}

template <template <typename, typename> class Trait, typename T, typename Class,
          bool Expected>
void check_binary_trait() {
  using base_t = std::integral_constant<bool, Expected>;
  STATIC_CHECK(std::is_base_of_v<base_t, Trait<T, Class>>);
  STATIC_CHECK(Trait<T, Class>::value == Expected);
}

/**
 * Checks the traits that the specification requires for a kernel property `P`
 * with key `K`, and that the `_v` variables agree with `::value`.
 */
template <typename P, typename K>
void check_kernel_property_traits() {
  check_unary_trait<khr::is_property, P, true>();
  check_unary_trait<khr::is_property_key, K, true>();
  check_unary_trait<khr::is_property_key_compile_time, K, true>();
  check_binary_trait<khr::is_property_for, P, kernel_tag, true>();
  check_binary_trait<khr::is_property_key_for, K, kernel_tag, true>();
  check_binary_trait<khr::is_property_or_list_for, P, kernel_tag, true>();

  // A property is not a key and a key is not a property.
  check_unary_trait<khr::is_property, K, false>();
  check_unary_trait<khr::is_property_key, P, false>();
  check_unary_trait<khr::is_property_key_compile_time, P, false>();

  STATIC_CHECK(khr::is_property_v<P> == khr::is_property<P>::value);
  STATIC_CHECK(khr::is_property_key_v<K> == khr::is_property_key<K>::value);
  STATIC_CHECK(khr::is_property_key_compile_time_v<K> ==
               khr::is_property_key_compile_time<K>::value);
  STATIC_CHECK(khr::is_property_for_v<P, kernel_tag> ==
               khr::is_property_for<P, kernel_tag>::value);
  STATIC_CHECK(khr::is_property_key_for_v<K, kernel_tag> ==
               khr::is_property_key_for<K, kernel_tag>::value);
  STATIC_CHECK(khr::is_property_or_list_for_v<P, kernel_tag> ==
               khr::is_property_or_list_for<P, kernel_tag>::value);

  // The kernel properties do not apply to the classes and tags of
  // sycl_khr_properties.
  check_binary_trait<khr::is_property_for, P, sycl::queue, false>();
  check_binary_trait<khr::is_property_key_for, K, sycl::queue, false>();
  check_binary_trait<khr::is_property_or_list_for, P, sycl::queue, false>();
  check_binary_trait<khr::is_property_for, P, prop::tag::reduction, false>();
  check_binary_trait<khr::is_property_key_for, K, prop::tag::reduction,
                     false>();

  STATIC_CHECK(std::is_trivially_copyable_v<P>);
}

/**
 * Checks that a properties list containing only `P` has the property, that
 * the property can be obtained through the static `get_property` overload of
 * a compile-time key in a constant expression, and that its type is `P`.
 */
template <typename P, typename K>
void check_in_properties_list() {
  using list_t = decltype(khr::properties{P{}});
  STATIC_CHECK(list_t::template has_property<K>());
  STATIC_CHECK(std::is_same_v<decltype(list_t::template get_property<K>()), P>);
  constexpr P copy = list_t::template get_property<K>();
  (void)copy;
  check_binary_trait<khr::is_property_or_list_for, list_t, kernel_tag, true>();
}

TEST_CASE("khr::property::tag::kernel is a complete empty class",
          "[khr_kernel_properties]") {
  STATIC_CHECK(std::is_class_v<kernel_tag>);
  // sizeof is ill-formed for an incomplete type.
  STATIC_CHECK(sizeof(kernel_tag) > 0);
  STATIC_CHECK(std::is_empty_v<kernel_tag>);
  STATIC_CHECK(std::is_trivially_default_constructible_v<kernel_tag>);
  STATIC_CHECK(std::is_trivially_copyable_v<kernel_tag>);
  constexpr kernel_tag tag{};
  (void)tag;
}

TEST_CASE("work_group_size has the specified traits and value",
          "[khr_kernel_properties][work_group_size]") {
  SECTION("1 dimension") {
    using p_t = wg_size_t<8>;
    check_kernel_property_traits<p_t, key::work_group_size>();
    check_in_properties_list<p_t, key::work_group_size>();
    STATIC_CHECK(std::is_same_v<decltype(p_t::value), const std::size_t[1]>);
    STATIC_CHECK(p_t::value[0] == 8);
  }
  SECTION("2 dimensions") {
    using p_t = wg_size_t<8, 4>;
    check_kernel_property_traits<p_t, key::work_group_size>();
    check_in_properties_list<p_t, key::work_group_size>();
    STATIC_CHECK(std::is_same_v<decltype(p_t::value), const std::size_t[2]>);
    STATIC_CHECK(p_t::value[0] == 8);
    STATIC_CHECK(p_t::value[1] == 4);
  }
  SECTION("3 dimensions") {
    using p_t = wg_size_t<8, 4, 2>;
    check_kernel_property_traits<p_t, key::work_group_size>();
    check_in_properties_list<p_t, key::work_group_size>();
    STATIC_CHECK(std::is_same_v<decltype(p_t::value), const std::size_t[3]>);
    STATIC_CHECK(p_t::value[0] == 8);
    STATIC_CHECK(p_t::value[1] == 4);
    STATIC_CHECK(p_t::value[2] == 2);
  }
  SECTION("the value is accessible through the variable template") {
    STATIC_CHECK(prop::work_group_size<16, 2>.value[0] == 16);
    STATIC_CHECK(prop::work_group_size<16, 2>.value[1] == 2);
  }
  SECTION("different values have different types") {
    STATIC_CHECK_FALSE(std::is_same_v<wg_size_t<8>, wg_size_t<4>>);
    STATIC_CHECK_FALSE(std::is_same_v<wg_size_t<8>, wg_size_t<8, 1>>);
    STATIC_CHECK_FALSE(std::is_same_v<wg_size_t<8, 4>, wg_size_t<4, 8>>);
  }
}

TEST_CASE("work_group_size_hint has the specified traits and value",
          "[khr_kernel_properties][work_group_size_hint]") {
  SECTION("1 dimension") {
    using p_t = wg_size_hint_t<8>;
    check_kernel_property_traits<p_t, key::work_group_size_hint>();
    check_in_properties_list<p_t, key::work_group_size_hint>();
    STATIC_CHECK(std::is_same_v<decltype(p_t::value), const std::size_t[1]>);
    STATIC_CHECK(p_t::value[0] == 8);
  }
  SECTION("2 dimensions") {
    using p_t = wg_size_hint_t<8, 4>;
    check_kernel_property_traits<p_t, key::work_group_size_hint>();
    check_in_properties_list<p_t, key::work_group_size_hint>();
    STATIC_CHECK(std::is_same_v<decltype(p_t::value), const std::size_t[2]>);
    STATIC_CHECK(p_t::value[0] == 8);
    STATIC_CHECK(p_t::value[1] == 4);
  }
  SECTION("3 dimensions") {
    using p_t = wg_size_hint_t<8, 4, 2>;
    check_kernel_property_traits<p_t, key::work_group_size_hint>();
    check_in_properties_list<p_t, key::work_group_size_hint>();
    STATIC_CHECK(std::is_same_v<decltype(p_t::value), const std::size_t[3]>);
    STATIC_CHECK(p_t::value[0] == 8);
    STATIC_CHECK(p_t::value[1] == 4);
    STATIC_CHECK(p_t::value[2] == 2);
  }
  SECTION("the value is accessible through the variable template") {
    STATIC_CHECK(prop::work_group_size_hint<16, 2>.value[0] == 16);
    STATIC_CHECK(prop::work_group_size_hint<16, 2>.value[1] == 2);
  }
  SECTION("different values have different types") {
    STATIC_CHECK_FALSE(std::is_same_v<wg_size_hint_t<8>, wg_size_hint_t<4>>);
    STATIC_CHECK_FALSE(std::is_same_v<wg_size_hint_t<8>, wg_size_hint_t<8, 1>>);
  }
  SECTION("work_group_size_hint is not work_group_size") {
    STATIC_CHECK_FALSE(std::is_same_v<wg_size_hint_t<8>, wg_size_t<8>>);
    STATIC_CHECK_FALSE(
        std::is_same_v<key::work_group_size_hint, key::work_group_size>);
  }
}

TEST_CASE("sub_group_size has the specified traits and value",
          "[khr_kernel_properties][sub_group_size]") {
  using p_t = sg_size_t<16>;
  check_kernel_property_traits<p_t, key::sub_group_size>();
  check_in_properties_list<p_t, key::sub_group_size>();
  STATIC_CHECK(std::is_same_v<decltype(p_t::value), const std::uint32_t>);
  STATIC_CHECK(p_t::value == 16);
  STATIC_CHECK(prop::sub_group_size<32>.value == 32);
  STATIC_CHECK_FALSE(std::is_same_v<sg_size_t<16>, sg_size_t<32>>);
}

TEST_CASE("properties lists of kernel properties", "[khr_kernel_properties]") {
  SECTION("all three properties in one list") {
    constexpr khr::properties props{prop::work_group_size<8, 4>,
                                    prop::work_group_size_hint<16>,
                                    prop::sub_group_size<16>};
    using list_t = std::remove_cv_t<decltype(props)>;
    check_binary_trait<khr::is_property_or_list_for, list_t, kernel_tag,
                       true>();
    check_binary_trait<khr::is_property_or_list_for, list_t, sycl::queue,
                       false>();
    STATIC_CHECK(props.has_property<key::work_group_size>());
    STATIC_CHECK(props.has_property<key::work_group_size_hint>());
    STATIC_CHECK(props.has_property<key::sub_group_size>());
    STATIC_CHECK(
        std::is_same_v<decltype(list_t::get_property<key::work_group_size>()),
                       wg_size_t<8, 4>>);
    STATIC_CHECK(std::is_same_v<
                 decltype(list_t::get_property<key::work_group_size_hint>()),
                 wg_size_hint_t<16>>);
    STATIC_CHECK(
        std::is_same_v<decltype(list_t::get_property<key::sub_group_size>()),
                       sg_size_t<16>>);
    STATIC_CHECK(list_t::get_property<key::work_group_size>().value[1] == 4);
    STATIC_CHECK(list_t::get_property<key::sub_group_size>().value == 16);
  }
  SECTION("a list without a kernel property does not have its key") {
    using list_t = decltype(khr::properties{prop::sub_group_size<16>});
    STATIC_CHECK_FALSE(list_t::has_property<key::work_group_size>());
    STATIC_CHECK_FALSE(list_t::has_property<key::work_group_size_hint>());
    STATIC_CHECK(list_t::has_property<key::sub_group_size>());
  }
  SECTION("the empty properties list applies to kernels") {
    check_binary_trait<khr::is_property_or_list_for, khr::empty_properties_t,
                       kernel_tag, true>();
  }
  SECTION("lists containing a property that is not for kernels") {
    using with_queue_t =
        decltype(khr::properties{prop::work_group_size<8>, prop::in_order{}});
    using queue_only_t = decltype(khr::properties{prop::in_order{}});
    check_binary_trait<khr::is_property_or_list_for, with_queue_t, kernel_tag,
                       false>();
    check_binary_trait<khr::is_property_or_list_for, queue_only_t, kernel_tag,
                       false>();
  }
}

TEST_CASE("properties of sycl_khr_properties do not apply to kernels",
          "[khr_kernel_properties]") {
  check_binary_trait<khr::is_property_for, prop::in_order, kernel_tag, false>();
  check_binary_trait<khr::is_property_for, prop::enable_profiling, kernel_tag,
                     false>();
  check_binary_trait<khr::is_property_for, prop::initialize_to_identity,
                     kernel_tag, false>();
  check_binary_trait<khr::is_property_key_for, key::in_order, kernel_tag,
                     false>();
  check_binary_trait<khr::is_property_key_for, key::enable_profiling,
                     kernel_tag, false>();
  check_binary_trait<khr::is_property_key_for, key::initialize_to_identity,
                     kernel_tag, false>();
  check_binary_trait<khr::is_property_or_list_for, not_a_property, kernel_tag,
                     false>();
  check_binary_trait<khr::is_property_or_list_for, kernel_tag, kernel_tag,
                     false>();
}

TEST_CASE("the queue constructors reject kernel properties",
          "[khr_kernel_properties]") {
  STATIC_CHECK_FALSE(std::is_constructible_v<sycl::queue, wg_size_t<8>>);
  STATIC_CHECK_FALSE(std::is_constructible_v<sycl::queue, wg_size_hint_t<8>>);
  STATIC_CHECK_FALSE(std::is_constructible_v<sycl::queue, sg_size_t<16>>);
  STATIC_CHECK_FALSE(
      std::is_constructible_v<sycl::queue, decltype(khr::properties{
                                               prop::work_group_size<8>})>);
}

}  // namespace khr_kernel_properties::tests
