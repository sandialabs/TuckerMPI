#ifndef IMPL_VIEW_CLONE_UTILS_HPP_
#define IMPL_VIEW_CLONE_UTILS_HPP_

#include <Kokkos_Core.hpp>

namespace Tucker{
namespace impl{

template <class ViewTypeFrom, class ViewTypeTo>
struct CopyFunctor {
  ViewTypeFrom m_view_from;
  ViewTypeTo m_view_to;

  CopyFunctor() = delete;

  CopyFunctor(const ViewTypeFrom view_from, const ViewTypeTo view_to)
      : m_view_from(view_from), m_view_to(view_to) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { m_view_to(i) = m_view_from(i); }
};

template <class ViewTypeFrom, class ViewTypeTo>
struct CopyFunctorRank2 {
  ViewTypeFrom m_view_from;
  ViewTypeTo m_view_to;

  CopyFunctorRank2() = delete;

  CopyFunctorRank2(const ViewTypeFrom view_from, const ViewTypeTo view_to)
      : m_view_from(view_from), m_view_to(view_to) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int k) const {
    const auto i    = k / m_view_from.extent(1);
    const auto j    = k % m_view_from.extent(1);
    m_view_to(i, j) = m_view_from(i, j);
  }
};

template <class ViewType>
auto create_deep_copyable_compatible_view_with_same_extent(ViewType view) {
  using view_value_type  = typename ViewType::value_type;
  using view_exespace    = typename ViewType::execution_space;
  const std::size_t ext0 = view.extent(0);
  if constexpr (ViewType::rank == 1) {
    using view_deep_copyable_t = Kokkos::View<view_value_type*, Kokkos::LayoutLeft, view_exespace>;
    return view_deep_copyable_t{"view_dc", ext0};
  } else {
    static_assert(ViewType::rank == 2, "Only rank 1 or 2 supported.");
    using view_deep_copyable_t = Kokkos::View<view_value_type**, Kokkos::LayoutLeft, view_exespace>;
    const std::size_t ext1     = view.extent(1);
    return view_deep_copyable_t{"view_dc", ext0, ext1};
  }
}

template <class ViewType>
auto create_deep_copyable_compatible_clone(ViewType view) {
  auto view_dc    = create_deep_copyable_compatible_view_with_same_extent(view);
  using view_dc_t = decltype(view_dc);
  if constexpr (ViewType::rank == 1) {
    CopyFunctor<ViewType, view_dc_t> F1(view, view_dc);
    Kokkos::parallel_for("copy", view.extent(0), F1);
  } else {
    static_assert(ViewType::rank == 2, "Only rank 1 or 2 supported.");
    CopyFunctorRank2<ViewType, view_dc_t> F1(view, view_dc);
    Kokkos::parallel_for("copy", view.extent(0) * view.extent(1), F1);
  }
  return view_dc;
}

} //end namespace impl
} //endm namespace Tucker
#endif  // IMPL_TUCKERONNODE_TTM_USING_HOST_BLAS_IMPL_HPP_
