
#ifndef IMPL_TUCKER_TUCKERTENSORSLICEHELPERS_HPP_
#define IMPL_TUCKER_TUCKERTENSORSLICEHELPERS_HPP_

#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Core.hpp>
#include <cuchar>

namespace Tucker{
namespace impl{

struct PerModeSliceInfo{
  std::size_t startIndex        = 0;
  std::size_t endIndexExclusive = 0;
  std::size_t extent0 = 0;
  std::size_t extent1 = 0;
};

template<
  class DataType1, class ...Props1,
  class DataType2, class ...Props2
>
void appendEigenvaluesAndUpdateSliceInfo(int mode,
					 Kokkos::View<DataType1, Props1...> & dest,
					 Kokkos::View<DataType2, Props2...> & src,
					 ::Tucker::impl::PerModeSliceInfo & sliceInfo)
{
  namespace KEX = Kokkos::Experimental;

  // constraints
  using dest_view =  Kokkos::View<DataType1, Props1...>;
  // both must be rank1, same mem space

  // preconditions
  assert(mode>=0);

  // use Kokkos::resize to preserve the current content of the view
  const std::size_t currentExt = dest.extent(0);
  Kokkos::resize(dest, currentExt + src.extent(0));

  // copy the data
  auto it0 = KEX::begin(dest);
  auto outItBegin = it0 + currentExt;
  using exespace = typename dest_view::execution_space;
  auto resIt = KEX::copy(exespace(), KEX::cbegin(src), KEX::cend(src), outItBegin);

  // update slicing info
  sliceInfo.startIndex = KEX::distance(it0, outItBegin);
  sliceInfo.endIndexExclusive = KEX::distance(it0, resIt);
}

template <class IteratorType, class ViewType>
struct CopyFactorData
{
  IteratorType outIt_;
  ViewType src_;
  CopyFactorData(IteratorType it, ViewType view) : outIt_(it), src_(view){}

  KOKKOS_FUNCTION void operator()(std::size_t k) const{
    const std::size_t nR = src_.extent(0);
    const std::size_t row = k % nR;
    const std::size_t col = k / nR;
    *(outIt_ + k) = src_(row, col);
  }
};

template<class DataType, class ...Props, class SourceViewType>
void appendFactorsAndUpdateSliceInfo(int mode,
				     Kokkos::View<DataType, Props...> & dest,
				     SourceViewType src,
				     ::Tucker::impl::PerModeSliceInfo & sliceInfo)
{
  namespace KEX = Kokkos::Experimental;

  // use Kokkos::resize to preserve the current content of the view
  const std::size_t currentExt = dest.extent(0);
  Kokkos::resize(dest, currentExt + src.size());

  // copy the data
  auto it0 = KEX::begin(dest);
  auto outItBegin = it0 + currentExt;
  Kokkos::parallel_for(src.size(), CopyFactorData(outItBegin, src));

  // update slicing info
  sliceInfo.startIndex = currentExt;
  sliceInfo.endIndexExclusive = currentExt + src.size();
  sliceInfo.extent0 = src.extent(0);
  sliceInfo.extent1 = src.extent(1);
}

}}
#endif  // IMPL_TUCKER_TUCKERTENSORSLICEHELPERS_HPP_
