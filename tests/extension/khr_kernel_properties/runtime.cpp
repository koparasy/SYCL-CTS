/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Runtime tests for the sycl_khr_kernel_properties extension: the effects of
//  embedding the khr::property::work_group_size,
//  khr::property::work_group_size_hint and khr::property::sub_group_size
//  properties in a kernel that is defined as a named function object. The
//  properties themselves are tested in kernel_properties.cpp. These tests need
//  a device, so they use the CTS helpers and <sycl/sycl.hpp>.
//
//  Each property must have the same effect as the corresponding kernel
//  attribute, including the effect on the kernel information queries. The
//  tests therefore compare the kernel information queries of a kernel with an
//  embedded property against those of an equivalent kernel decorated with the
//  attribute.
//
//  The implementation determines the properties from the return type of the
//  get() member function and does not call it. Every get() member function in
//  this file is declared but not defined, so calling it is a link error.
//
*******************************************************************************/

#include "../../../util/sycl_exceptions.h"
#include "../../common/common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace khr_kernel_properties::tests {

namespace khr = sycl::khr;
namespace prop = sycl::khr::property;
using kernel_tag = sycl::khr::property::tag::kernel;

template <std::size_t... Dims>
using wg_size_t = std::remove_cv_t<decltype(prop::work_group_size<Dims...>)>;
template <std::size_t... Dims>
using wg_size_hint_t =
    std::remove_cv_t<decltype(prop::work_group_size_hint<Dims...>)>;
template <std::uint32_t Size>
using sg_size_t = std::remove_cv_t<decltype(prop::sub_group_size<Size>)>;

/**
 * The sub-group sizes the tests try. Tests that need a supported size skip
 * the sizes the device does not support, and are skipped if it supports none.
 */
constexpr std::array<std::uint32_t, 7> sub_group_size_candidates{1,  2,  4, 8,
                                                                 16, 32, 64};

using out_acc_t = sycl::accessor<std::size_t, 1, sycl::access_mode::write>;

template <int Dims>
sycl::range<Dims> make_range(const std::array<std::size_t, Dims>& sizes) {
  if constexpr (Dims == 1) {
    return sycl::range<1>{sizes[0]};
  } else if constexpr (Dims == 2) {
    return sycl::range<2>{sizes[0], sizes[1]};
  } else {
    return sycl::range<3>{sizes[0], sizes[1], sizes[2]};
  }
}

template <int Dims>
sycl::range<Dims> ones() {
  return make_range<Dims>([] {
    std::array<std::size_t, Dims> a{};
    a.fill(1);
    return a;
  }());
}

/**
 * Whether the device supports a work-group of the given size, as required by
 * the reqd_work_group_size attribute.
 */
template <int Dims>
bool supports_work_group_size(const sycl::device& dev,
                              const std::array<std::size_t, Dims>& sizes) {
  const auto max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
  const auto max_items =
      dev.get_info<sycl::info::device::max_work_item_sizes<Dims>>();
  std::size_t product = 1;
  for (int i = 0; i < Dims; ++i) {
    if (sizes[i] > max_items[i]) return false;
    if (sizes[i] > max_wg / product) return false;
    product *= sizes[i];
  }
  return product <= max_wg;
}

bool supports_sub_group_size(const sycl::device& dev, std::uint32_t size) {
  const auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  return std::find(sizes.begin(), sizes.end(), std::size_t{size}) !=
         sizes.end();
}

template <typename KernelName>
sycl::kernel get_kernel(const sycl::queue& q) {
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q.get_context(), {q.get_device()}, {sycl::get_kernel_id<KernelName>()});
  return bundle.template get_kernel<KernelName>();
}

template <typename KernelName>
sycl::range<3> compile_work_group_size(const sycl::queue& q) {
  return get_kernel<KernelName>(q)
      .template get_info<
          sycl::info::kernel_device_specific::compile_work_group_size>(
          q.get_device());
}

template <typename KernelName>
std::uint32_t compile_sub_group_size(const sycl::queue& q) {
  return get_kernel<KernelName>(q)
      .template get_info<
          sycl::info::kernel_device_specific::compile_sub_group_size>(
          q.get_device());
}

/** Writes the local range seen by the first work-item to `out`. */
template <int Dims>
void write_local_range(const sycl::nd_item<Dims>& it, const out_acc_t& out) {
  if (it.get_global_linear_id() == 0) {
    for (int i = 0; i < Dims; ++i) out[i] = it.get_local_range(i);
  }
}

/** Writes the sub-group size seen by the first work-item to `out[0]`. */
template <int Dims>
void write_sub_group_size(const sycl::nd_item<Dims>& it, const out_acc_t& out) {
  if (it.get_global_linear_id() == 0) {
    out[0] = it.get_sub_group().get_max_local_range()[0];
  }
}

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

/** Embeds work_group_size<Dims...> as a bare property. */
template <std::size_t... Dims>
struct wg_size_kernel {
  static constexpr int dims = sizeof...(Dims);
  out_acc_t out;
  void operator()(sycl::nd_item<dims> it) const { write_local_range(it, out); }
  wg_size_t<Dims...> get(kernel_tag) const;
};

/** Embeds work_group_size<Dims...> and has no kernel arguments. */
template <std::size_t... Dims>
struct wg_size_empty_kernel {
  static constexpr int dims = sizeof...(Dims);
  void operator()(sycl::nd_item<dims>) const {}
  wg_size_t<Dims...> get(kernel_tag) const;
};

/** The equivalent kernels decorated with reqd_work_group_size. */
template <std::size_t N0>
struct wg_size_attr_kernel_1 {
  [[sycl::reqd_work_group_size(N0)]] void operator()(sycl::nd_item<1>) const {}
};
template <std::size_t N0, std::size_t N1>
struct wg_size_attr_kernel_2 {
  [[sycl::reqd_work_group_size(N0, N1)]] void operator()(
      sycl::nd_item<2>) const {}
};
template <std::size_t N0, std::size_t N1, std::size_t N2>
struct wg_size_attr_kernel_3 {
  [[sycl::reqd_work_group_size(N0, N1, N2)]] void operator()(
      sycl::nd_item<3>) const {}
};

template <std::size_t... Dims>
struct wg_size_attr_kernel_selector;
template <std::size_t N0>
struct wg_size_attr_kernel_selector<N0> {
  using type = wg_size_attr_kernel_1<N0>;
};
template <std::size_t N0, std::size_t N1>
struct wg_size_attr_kernel_selector<N0, N1> {
  using type = wg_size_attr_kernel_2<N0, N1>;
};
template <std::size_t N0, std::size_t N1, std::size_t N2>
struct wg_size_attr_kernel_selector<N0, N1, N2> {
  using type = wg_size_attr_kernel_3<N0, N1, N2>;
};
template <std::size_t... Dims>
using wg_size_attr_kernel =
    typename wg_size_attr_kernel_selector<Dims...>::type;

/**
 * Embeds work_group_size<N0> in a single-element properties list and also
 * decorates the call operator with the same reqd_work_group_size attribute.
 */
template <std::size_t N0>
struct wg_size_and_attr_kernel {
  [[sycl::reqd_work_group_size(N0)]] void operator()(sycl::nd_item<1>) const {}
  decltype(khr::properties{prop::work_group_size<N0>}) get(kernel_tag) const;
};

/** Embeds work_group_size_hint<Dims...> as a bare property. */
template <std::size_t... Dims>
struct wg_size_hint_kernel {
  static constexpr int dims = sizeof...(Dims);
  out_acc_t out;
  void operator()(sycl::nd_item<dims> it) const { write_local_range(it, out); }
  wg_size_hint_t<Dims...> get(kernel_tag) const;
};

/** Embeds sub_group_size<Size> as a bare property. */
template <std::uint32_t Size>
struct sg_size_kernel {
  out_acc_t out;
  void operator()(sycl::nd_item<1> it) const { write_sub_group_size(it, out); }
  sg_size_t<Size> get(kernel_tag) const;
};

/** The equivalent kernel decorated with reqd_sub_group_size. */
template <std::uint32_t Size>
struct sg_size_attr_kernel {
  [[sycl::reqd_sub_group_size(Size)]] void operator()(sycl::nd_item<1>) const {}
};

/**
 * Embeds sub_group_size<Size> in a single-element properties list and also
 * decorates the call operator with the same reqd_sub_group_size attribute.
 */
template <std::uint32_t Size>
struct sg_size_and_attr_kernel {
  out_acc_t out;
  [[sycl::reqd_sub_group_size(Size)]] void operator()(
      sycl::nd_item<1> it) const {
    write_sub_group_size(it, out);
  }
  decltype(khr::properties{prop::sub_group_size<Size>}) get(kernel_tag) const;
};

/**
 * Embeds sub_group_size<Size> and has two function call operators. The
 * embedded properties apply to both of them.
 */
template <std::uint32_t Size>
struct sg_size_two_operators_kernel {
  out_acc_t out;
  void operator()(sycl::nd_item<1> it) const { write_sub_group_size(it, out); }
  void operator()(sycl::nd_item<2> it) const {
    if (it.get_global_linear_id() == 0) {
      out[0] = it.get_sub_group().get_max_local_range()[0];
    }
  }
  sg_size_t<Size> get(kernel_tag) const;
};
template <std::uint32_t Size>
class sg_size_two_operators_1d;
template <std::uint32_t Size>
class sg_size_two_operators_2d;

/**
 * Embeds work_group_size<N> and has an nd_item and a group function call
 * operator. The embedded properties apply to both of them.
 */
template <std::size_t N>
struct wg_size_two_operators_kernel {
  out_acc_t out;
  void operator()(sycl::nd_item<1> it) const { write_local_range(it, out); }
  void operator()(sycl::group<1> g) const {
    if (g.get_group_linear_id() == 0) out[0] = g.get_local_range(0);
  }
  wg_size_t<N> get(kernel_tag) const;
};
template <std::size_t N>
class wg_size_two_operators_nd_item;
template <std::size_t N>
class wg_size_two_operators_group;

/** Embeds the empty properties list. */
struct empty_list_kernel {
  void operator()(sycl::nd_item<1>) const {}
  khr::empty_properties_t get(kernel_tag) const;
};

/** The same kernel without embedded properties. */
struct no_properties_kernel {
  void operator()(sycl::nd_item<1>) const {}
};

/** Embeds all three properties in one properties list. */
template <std::size_t WG, std::uint32_t SG>
struct all_properties_kernel {
  out_acc_t out;
  void operator()(sycl::nd_item<1> it) const {
    if (it.get_global_linear_id() == 0) {
      out[0] = it.get_local_range(0);
      out[1] = it.get_sub_group().get_max_local_range()[0];
    }
  }
  decltype(khr::properties{prop::work_group_size<WG>,
                           prop::work_group_size_hint<WG>,
                           prop::sub_group_size<SG>}) get(kernel_tag) const;
};

// ----------------------------------------------------------------------------
// work_group_size
// ----------------------------------------------------------------------------

template <std::size_t... Dims>
void check_work_group_size() {
  constexpr int dims = sizeof...(Dims);
  constexpr std::array<std::size_t, dims> sizes{Dims...};
  using kernel_t = wg_size_kernel<Dims...>;
  using empty_kernel_t = wg_size_empty_kernel<Dims...>;
  using attr_kernel_t = wg_size_attr_kernel<Dims...>;

  auto q = sycl_cts::util::get_cts_object::queue();
  const sycl::device dev = q.get_device();
  const sycl::range<dims> local = make_range<dims>(sizes);

  // Reqd work-group sizes are optional kernel features, so the kernel is
  // compatible with the device exactly when the attribute kernel is.
  CHECK(sycl::is_compatible<empty_kernel_t>(dev) ==
        sycl::is_compatible<attr_kernel_t>(dev));

  if (!supports_work_group_size<dims>(dev, sizes)) {
    INFO("The device does not support the work-group size");
    CHECK_FALSE(sycl::is_compatible<empty_kernel_t>(dev));
    // A single work-group, so that the nd_range is otherwise valid.
    const sycl::nd_range<dims> one_group{local, local};
    CHECK_THROWS_MATCHES(
        q.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(one_group, empty_kernel_t{});
        }),
        sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::kernel_not_supported));
    CHECK_THROWS_MATCHES(
        q.parallel_for(one_group, empty_kernel_t{}), sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::kernel_not_supported));
    return;
  }

  const sycl::range<dims> global = local * 2;
  const sycl::nd_range<dims> ndr{global, local};

  SECTION("the kernel runs with the required work-group size") {
    std::vector<std::size_t> result(dims, 0);
    {
      sycl::buffer<std::size_t, 1> buf{result.data(), sycl::range<1>{dims}};
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(ndr, kernel_t{out_acc_t{buf, cgh}});
      });
    }
    for (int i = 0; i < dims; ++i) {
      INFO("dimension " << i);
      CHECK(result[i] == sizes[i]);
    }
  }

  SECTION("the kernel runs through the queue shortcut") {
    q.parallel_for(ndr, empty_kernel_t{}).wait_and_throw();
  }

  SECTION("a different work-group size throws errc::nd_range") {
    // Every required size used by the tests has an extent greater than 1.
    const sycl::nd_range<dims> bad_ndr{global, ones<dims>()};
    CHECK_THROWS_MATCHES(
        q.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(bad_ndr, empty_kernel_t{});
        }),
        sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::nd_range));
    CHECK_THROWS_MATCHES(
        q.parallel_for(bad_ndr, empty_kernel_t{}), sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::nd_range));
  }

  SECTION(
      "compile_work_group_size matches the reqd_work_group_size attribute") {
    q.parallel_for(ndr, attr_kernel_t{}).wait_and_throw();
    q.parallel_for(ndr, empty_kernel_t{}).wait_and_throw();
    const sycl::range<3> expected = compile_work_group_size<attr_kernel_t>(q);
    const sycl::range<3> actual = compile_work_group_size<empty_kernel_t>(q);
    CHECK(actual == expected);
  }
}

TEST_CASE("an embedded work_group_size property requires the work-group size",
          "[khr_kernel_properties][work_group_size]") {
  SECTION("1 dimension") { check_work_group_size<8>(); }
  SECTION("2 dimensions") { check_work_group_size<8, 4>(); }
  SECTION("3 dimensions") { check_work_group_size<4, 2, 2>(); }
}

TEST_CASE(
    "an embedded work_group_size property that the device does not support "
    "throws errc::kernel_not_supported",
    "[khr_kernel_properties][work_group_size]") {
  // 2^32 on 64-bit hosts, otherwise the largest std::size_t. static_cast
  // avoids the narrowing that list-initialization would diagnose on 32-bit
  // hosts.
  constexpr std::size_t n0 = sizeof(std::size_t) == 8
                                 ? static_cast<std::size_t>(4294967296ull)
                                 : std::numeric_limits<std::size_t>::max();
  check_work_group_size<n0>();
}

TEST_CASE(
    "a kernel may combine an embedded work_group_size property with the "
    "same reqd_work_group_size attribute",
    "[khr_kernel_properties][work_group_size]") {
  constexpr std::size_t N = 8;
  using kernel_t = wg_size_and_attr_kernel<N>;
  using attr_kernel_t = wg_size_attr_kernel_1<N>;
  auto q = sycl_cts::util::get_cts_object::queue();
  if (!supports_work_group_size<1>(q.get_device(), {N})) {
    SKIP("The device does not support the work-group size");
  }
  const sycl::nd_range<1> ndr{{2 * N}, {N}};
  q.parallel_for(ndr, kernel_t{}).wait_and_throw();
  q.parallel_for(ndr, attr_kernel_t{}).wait_and_throw();
  CHECK(compile_work_group_size<kernel_t>(q) ==
        compile_work_group_size<attr_kernel_t>(q));
  CHECK_THROWS_MATCHES(
      q.parallel_for(sycl::nd_range<1>{{2 * N}, {1}}, kernel_t{}),
      sycl::exception, sycl_cts::util::equals_exception(sycl::errc::nd_range));
}

// ----------------------------------------------------------------------------
// work_group_size_hint
// ----------------------------------------------------------------------------

template <std::size_t... Dims>
void check_work_group_size_hint() {
  constexpr int dims = sizeof...(Dims);
  constexpr std::array<std::size_t, dims> sizes{Dims...};
  using kernel_t = wg_size_hint_kernel<Dims...>;

  auto q = sycl_cts::util::get_cts_object::queue();
  const sycl::range<dims> hinted = make_range<dims>(sizes);

  // The hint does not restrict the work-group size the kernel is launched
  // with. A work-group of one work-item is valid on every device; the hinted
  // size is only launched if the device supports it.
  std::vector<sycl::range<dims>> local_sizes{ones<dims>()};
  if (supports_work_group_size<dims>(q.get_device(), sizes)) {
    local_sizes.push_back(hinted);
  } else {
    WARN("The device does not support the hinted work-group size");
  }

  for (const sycl::range<dims>& local : local_sizes) {
    std::vector<std::size_t> result(dims, 0);
    {
      sycl::buffer<std::size_t, 1> buf{result.data(), sycl::range<1>{dims}};
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<dims>{hinted * 2, local},
                         kernel_t{out_acc_t{buf, cgh}});
      });
    }
    for (int i = 0; i < dims; ++i) {
      INFO("dimension " << i);
      CHECK(result[i] == local[i]);
    }
  }
}

TEST_CASE(
    "an embedded work_group_size_hint property does not restrict the "
    "work-group size",
    "[khr_kernel_properties][work_group_size_hint]") {
  SECTION("1 dimension") { check_work_group_size_hint<8>(); }
  SECTION("2 dimensions") { check_work_group_size_hint<8, 4>(); }
  SECTION("3 dimensions") { check_work_group_size_hint<4, 2, 2>(); }
}

// ----------------------------------------------------------------------------
// sub_group_size
// ----------------------------------------------------------------------------

/**
 * Calls `f.template operator()<Size>()` for each sub-group size candidate.
 * Each call returns whether the device supports `Size`. Returns the number of
 * supported candidates.
 */
template <typename F, std::size_t... I>
int for_each_sub_group_size(F&& f, std::index_sequence<I...>) {
  return (0 + ... + int(f.template operator()<sub_group_size_candidates[I]>()));
}
template <typename F>
int for_each_sub_group_size(F&& f) {
  return for_each_sub_group_size(
      std::forward<F>(f),
      std::make_index_sequence<sub_group_size_candidates.size()>{});
}

/** Skips the test case if the device supports no candidate sub-group size. */
#define KHR_KERNEL_PROPERTIES_SKIP_IF_NONE(count)                        \
  do {                                                                   \
    if ((count) == 0) {                                                  \
      SKIP("The device supports none of the candidate sub-group sizes"); \
    }                                                                    \
  } while (0)

/**
 * Runs `Kernel` in a 1-dimensional nd_range of two work-groups of `local`
 * work-items and returns the value it writes to `out[0]`.
 */
template <typename Kernel>
std::size_t run_and_read(sycl::queue& q, std::size_t local) {
  std::size_t result = 0;
  {
    sycl::buffer<std::size_t, 1> buf{&result, sycl::range<1>{1}};
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>{{2 * local}, {local}},
                       Kernel{out_acc_t{buf, cgh}});
    });
  }
  return result;
}

/**
 * Checks that submitting `Kernel` throws errc::kernel_not_supported. The
 * nd_range has a single work-item, so that the sub-group size is the only
 * reason for the kernel to be rejected.
 */
template <typename Kernel>
void check_kernel_not_supported(sycl::queue& q) {
  CHECK_FALSE(sycl::is_compatible<Kernel>(q.get_device()));
  sycl::buffer<std::size_t, 1> buf{sycl::range<1>{2}};
  CHECK_THROWS_MATCHES(q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>{{1}, {1}}, Kernel{out_acc_t{buf, cgh}});
  }),
                       sycl::exception,
                       sycl_cts::util::equals_exception(
                           sycl::errc::kernel_not_supported));
}

struct check_sub_group_size {
  template <std::uint32_t Size>
  bool operator()() const {
    using kernel_t = sg_size_kernel<Size>;
    using attr_kernel_t = sg_size_attr_kernel<Size>;
    auto q = sycl_cts::util::get_cts_object::queue();
    INFO("sub-group size " << Size);
    const sycl::nd_range<1> ndr{{2 * Size}, {Size}};

    CHECK(sycl::is_compatible<kernel_t>(q.get_device()) ==
          sycl::is_compatible<attr_kernel_t>(q.get_device()));

    if (!supports_sub_group_size(q.get_device(), Size)) {
      check_kernel_not_supported<kernel_t>(q);
      return false;
    }

    CHECK(run_and_read<kernel_t>(q, Size) == Size);
    q.parallel_for(ndr, attr_kernel_t{}).wait_and_throw();
    CHECK(compile_sub_group_size<kernel_t>(q) ==
          compile_sub_group_size<attr_kernel_t>(q));
    CHECK(compile_sub_group_size<kernel_t>(q) == Size);
    return true;
  }
};

TEST_CASE("an embedded sub_group_size property requires the sub-group size",
          "[khr_kernel_properties][sub_group_size]") {
  KHR_KERNEL_PROPERTIES_SKIP_IF_NONE(
      for_each_sub_group_size(check_sub_group_size{}));
}

TEST_CASE(
    "an embedded sub_group_size property that the device does not support "
    "throws errc::kernel_not_supported",
    "[khr_kernel_properties][sub_group_size]") {
  // No device supports a sub-group size that is not a power of two this
  // large; if one does, the kernel must run with it.
  check_sub_group_size{}.operator()<4099>();
}

struct check_sub_group_size_and_attribute {
  template <std::uint32_t Size>
  bool operator()() const {
    using kernel_t = sg_size_and_attr_kernel<Size>;
    auto q = sycl_cts::util::get_cts_object::queue();
    INFO("sub-group size " << Size);
    if (!supports_sub_group_size(q.get_device(), Size)) {
      check_kernel_not_supported<kernel_t>(q);
      return false;
    }
    CHECK(run_and_read<kernel_t>(q, Size) == Size);
    CHECK(compile_sub_group_size<kernel_t>(q) == Size);
    return true;
  }
};

TEST_CASE(
    "a kernel may combine an embedded sub_group_size property with the same "
    "reqd_sub_group_size attribute",
    "[khr_kernel_properties][sub_group_size]") {
  KHR_KERNEL_PROPERTIES_SKIP_IF_NONE(
      for_each_sub_group_size(check_sub_group_size_and_attribute{}));
}

struct check_two_operators {
  template <std::uint32_t Size>
  bool operator()() const {
    using kernel_t = sg_size_two_operators_kernel<Size>;
    using name_1d = sg_size_two_operators_1d<Size>;
    using name_2d = sg_size_two_operators_2d<Size>;
    auto q = sycl_cts::util::get_cts_object::queue();
    if (!supports_sub_group_size(q.get_device(), Size)) return false;
    INFO("sub-group size " << Size);

    std::size_t result_1d = 0;
    std::size_t result_2d = 0;
    {
      sycl::buffer<std::size_t, 1> buf_1d{&result_1d, sycl::range<1>{1}};
      sycl::buffer<std::size_t, 1> buf_2d{&result_2d, sycl::range<1>{1}};
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<name_1d>(sycl::nd_range<1>{{2 * Size}, {Size}},
                                  kernel_t{out_acc_t{buf_1d, cgh}});
      });
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<name_2d>(sycl::nd_range<2>{{2, 2 * Size}, {1, Size}},
                                  kernel_t{out_acc_t{buf_2d, cgh}});
      });
    }
    CHECK(result_1d == Size);
    CHECK(result_2d == Size);
    CHECK(compile_sub_group_size<name_1d>(q) == Size);
    CHECK(compile_sub_group_size<name_2d>(q) == Size);
    return true;
  }
};

TEST_CASE(
    "an embedded sub_group_size property applies to every function call "
    "operator",
    "[khr_kernel_properties][sub_group_size]") {
  KHR_KERNEL_PROPERTIES_SKIP_IF_NONE(
      for_each_sub_group_size(check_two_operators{}));
}

/**
 * Launches the nd_item operator of `wg_size_two_operators_kernel<N>` through
 * parallel_for and its group operator through parallel_for_work_group, and
 * checks that both run with work-groups of `N` work-items and report the
 * compile_work_group_size of the reqd_work_group_size attribute.
 */
template <std::size_t N>
void check_wg_size_two_operators(sycl::queue& q) {
  using kernel_t = wg_size_two_operators_kernel<N>;
  using name_nd_item = wg_size_two_operators_nd_item<N>;
  using name_group = wg_size_two_operators_group<N>;
  using attr_kernel_t = wg_size_attr_kernel_1<N>;
  INFO("work-group size " << N);

  std::size_t result_nd_item = 0;
  std::size_t result_group = 0;
  {
    sycl::buffer<std::size_t, 1> buf_nd_item{&result_nd_item,
                                             sycl::range<1>{1}};
    sycl::buffer<std::size_t, 1> buf_group{&result_group, sycl::range<1>{1}};
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<name_nd_item>(sycl::nd_range<1>{{2 * N}, {N}},
                                     kernel_t{out_acc_t{buf_nd_item, cgh}});
    });
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for_work_group<name_group>(
          sycl::range<1>{2}, sycl::range<1>{N},
          kernel_t{out_acc_t{buf_group, cgh}});
    });
  }
  CHECK(result_nd_item == N);
  CHECK(result_group == N);

  q.parallel_for(sycl::nd_range<1>{{2 * N}, {N}}, attr_kernel_t{})
      .wait_and_throw();
  const sycl::range<3> expected = compile_work_group_size<attr_kernel_t>(q);
  CHECK(compile_work_group_size<name_nd_item>(q) == expected);
  CHECK(compile_work_group_size<name_group>(q) == expected);

  if constexpr (N > 1) {
    sycl::buffer<std::size_t, 1> buf{sycl::range<1>{1}};
    CHECK_THROWS_MATCHES(
        q.submit([&](sycl::handler& cgh) {
          cgh.parallel_for<name_nd_item>(sycl::nd_range<1>{{2 * N}, {1}},
                                         kernel_t{out_acc_t{buf, cgh}});
        }),
        sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::nd_range));
  }
}

TEST_CASE(
    "an embedded work_group_size property applies to every function call "
    "operator",
    "[khr_kernel_properties][work_group_size]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  // Every device supports a work-group of one work-item.
  check_wg_size_two_operators<1>(q);
  if (supports_work_group_size<1>(q.get_device(), {8})) {
    check_wg_size_two_operators<8>(q);
  } else {
    WARN("The device does not support a work-group size of 8");
  }
}

// ----------------------------------------------------------------------------
// Properties lists
// ----------------------------------------------------------------------------

struct check_all_properties {
  template <std::uint32_t Size>
  bool operator()() const {
    constexpr std::size_t WG = 64;
    using kernel_t = all_properties_kernel<WG, Size>;
    using wg_attr_kernel_t = wg_size_attr_kernel_1<WG>;
    auto q = sycl_cts::util::get_cts_object::queue();
    const sycl::device dev = q.get_device();
    // A work-group of WG work-items holds WG / Size sub-groups, which the
    // device may not support for this sub-group size.
    if (!supports_sub_group_size(dev, Size) ||
        !supports_work_group_size<1>(dev, {WG}) || Size > WG ||
        WG / Size > dev.get_info<sycl::info::device::max_num_sub_groups>()) {
      return false;
    }
    INFO("sub-group size " << Size);

    std::array<std::size_t, 2> result{0, 0};
    {
      sycl::buffer<std::size_t, 1> buf{result.data(), sycl::range<1>{2}};
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>{{2 * WG}, {WG}},
                         kernel_t{out_acc_t{buf, cgh}});
      });
    }
    CHECK(result[0] == WG);
    CHECK(result[1] == Size);

    q.parallel_for(sycl::nd_range<1>{{2 * WG}, {WG}}, wg_attr_kernel_t{})
        .wait_and_throw();
    CHECK(compile_work_group_size<kernel_t>(q) ==
          compile_work_group_size<wg_attr_kernel_t>(q));
    CHECK(compile_sub_group_size<kernel_t>(q) == Size);

    sycl::buffer<std::size_t, 1> buf{sycl::range<1>{2}};
    CHECK_THROWS_MATCHES(
        q.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>{{2 * WG}, {WG / 2}},
                           kernel_t{out_acc_t{buf, cgh}});
        }),
        sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::nd_range));
    return true;
  }
};

TEST_CASE("a properties list embeds all of its properties",
          "[khr_kernel_properties]") {
  KHR_KERNEL_PROPERTIES_SKIP_IF_NONE(
      for_each_sub_group_size(check_all_properties{}));
}

TEST_CASE("embedding the empty properties list has no effect",
          "[khr_kernel_properties]") {
  // is_property_or_list_for is true for the empty properties list, so get()
  // may return it. The kernel must then behave as one without properties.
  auto q = sycl_cts::util::get_cts_object::queue();
  const sycl::nd_range<1> ndr{{2}, {1}};
  q.parallel_for(ndr, empty_list_kernel{}).wait_and_throw();
  q.parallel_for(ndr, no_properties_kernel{}).wait_and_throw();
  CHECK(compile_work_group_size<empty_list_kernel>(q) ==
        compile_work_group_size<no_properties_kernel>(q));
  CHECK(compile_sub_group_size<empty_list_kernel>(q) ==
        compile_sub_group_size<no_properties_kernel>(q));
}

}  // namespace khr_kernel_properties::tests
