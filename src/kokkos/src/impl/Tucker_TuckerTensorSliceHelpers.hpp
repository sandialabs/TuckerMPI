
#ifndef TUCKER_TUCKERTENSORSLICEHELPERS_HPP_
#define TUCKER_TUCKERTENSORSLICEHELPERS_HPP_

namespace Tucker{
namespace impl{

struct PerModeSliceInfo{
  std::size_t startIndex        = 0;
  std::size_t endIndexExclusive = 0;
  std::size_t extent0 = 0;
  std::size_t extent1 = 0;
};

}}
#endif
