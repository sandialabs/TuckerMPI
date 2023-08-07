#ifndef TUCKER_KOKKOS_MPI_PROD_IMPL_HPP_
#define TUCKER_KOKKOS_MPI_PROD_IMPL_HPP_

namespace TuckerMpi{
namespace impl{

template<class ViewT>
size_t prod(const ViewT & sz,
	    const int low, const int high,
	    const int defaultReturnVal = -1)
{
  if(low < 0 || high >= sz.extent(0)) {
    // std::cerr << "ERROR: prod(" << low << "," << high
    // 	      << ") is invalid because indices must be in the range [0,"
    // 	      << sz.extent(0) << ").  Returning " << defaultReturnVal << std::endl;
    return defaultReturnVal;
  }

  if(low > high) { return defaultReturnVal; }
  size_t result = 1;
  for(int j = low; j <= high; j++){ result *= sz[j]; }
  return result;
}

}}
#endif
