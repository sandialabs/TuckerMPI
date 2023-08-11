
#ifndef IMPL_TUCKER_STDVEC_VIEW_CONVERSION_HELPERS_HPP_
#define IMPL_TUCKER_STDVEC_VIEW_CONVERSION_HELPERS_HPP_

#include <Kokkos_Core.hpp>
#include <vector>

namespace Tucker{
namespace impl{

template <class T, class ...Properties>
void copy_stdvec_to_view(const std::vector<T> & from,
			 Kokkos::View<T*, Properties...> & to)
{
  assert(from.size() == to.extent(0));
  using view_t = Kokkos::View<T*, Properties...>;
  static_assert(
		(std::is_same<typename view_t::array_layout,
		 Kokkos::LayoutRight>::value ||
		 std::is_same<typename view_t::array_layout,
		 Kokkos::LayoutLeft>::value));

  auto to_h = Kokkos::create_mirror(to);
  for (std::size_t i=0; i<from.size(); ++i){
    to_h(i) = from[i];
  }
  Kokkos::deep_copy(to, to_h);
}

template <class T, class ...Properties>
void copy_view_to_stdvec(const Kokkos::View<T*, Properties...> & from,
			 std::vector<T> & to)
{
  using view_t = Kokkos::View<T*, Properties...>;
  static_assert(
		(std::is_same<typename view_t::array_layout,
		 Kokkos::LayoutRight>::value ||
		 std::is_same<typename view_t::array_layout,
		 Kokkos::LayoutLeft>::value));

  if (to.size() != from.extent(0)){
    to.resize(from.extent(0));
  }

  using mem_space = typename view_t::memory_space;
  if constexpr(Kokkos::SpaceAccessibility<Kokkos::HostSpace, mem_space>::accessible){
    for (std::size_t i=0; i<to.size(); ++i){ to[i] = from[i]; }
  }
  else{
    auto from_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), from);
    for (std::size_t i=0; i<to.size(); ++i){ to[i] = from_h[i]; }
  }
}

template <class T, class ...Properties>
auto create_stdvec_from_view(const Kokkos::View<T*, Properties...> & from)
{
  using view_t = Kokkos::View<T*, Properties...>;
  static_assert(
		(std::is_same<typename view_t::array_layout,
		 Kokkos::LayoutRight>::value ||
		 std::is_same<typename view_t::array_layout,
		 Kokkos::LayoutLeft>::value));

  using value_type = typename view_t::non_const_value_type;
  std::vector<value_type> v(from.extent(0));
  auto from_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), from);
  for (std::size_t i=0; i<from.extent(0); ++i){
    v[i] = from_h[i];
  }
  return v;
}

template <class T, class ...Properties>
void copy_view_to_stdvec(const Kokkos::View<T**, Properties...> & from,
			 std::vector<T> & to)
{
  using view_t = Kokkos::View<T**, Properties...>;
  static_assert(std::is_same<typename view_t::array_layout, Kokkos::LayoutLeft>::value);

  auto from_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), from);
  std::size_t k = 0;
  for (std::size_t j=0; j<from.extent(1); ++j){
    for (std::size_t i=0; i<from.extent(0); ++i){
      to[k] = from_h(i,j);
    }
  }
}

}}
#endif  // IMPL_TUCKER_STDVEC_VIEW_CONVERSION_HELPERS_HPP_
