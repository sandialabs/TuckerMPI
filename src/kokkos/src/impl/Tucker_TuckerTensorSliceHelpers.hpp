
#ifndef TUCKER_TUCKERTENSORSLICEHELPERS_HPP_
#define TUCKER_TUCKERTENSORSLICEHELPERS_HPP_

namespace Tucker{
namespace impl{

struct PerModeSliceInfo{
  std::size_t eigvalsStartIndex        = 0;
  std::size_t eigvalsEndIndexExclusive = 0;
  std::size_t factorsStartIndex        = 0;
  std::size_t factorsEndIndexExclusive = 0;
  std::size_t factorsExtent0           = 0;
  std::size_t factorsExtent1	       = 0;
};

}}
#endif
