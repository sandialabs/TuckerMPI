#ifndef CORE_TENSOR_TRUNCATOR_HPP_
#define CORE_TENSOR_TRUNCATOR_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include <variant>

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class ...Properties>
std::size_t count_eigvals_using_threshold(Kokkos::View<ScalarType*, Properties...> eigvals,
					  const ScalarType thresh)
{
  using eigvals_view_type = Kokkos::View<ScalarType*, Properties...>;
  using mem_space = typename eigvals_view_type::memory_space;
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, mem_space>::accessible,
		"count_eigvals_using_threshold: view must be accessible on host");

  std::size_t nrows = eigvals.extent(0);
  std::size_t numEvecs = nrows;
  ScalarType sum = 0;
  for(std::size_t i=nrows-1; i>=0; i--) {
    sum += std::abs(eigvals[i]);
    if(sum > thresh) {
      break;
    }
    numEvecs--;
  }
  return numEvecs;
}

} //end namespace impl

template <class ScalarType, class ...Properties>
auto create_core_tensor_truncator(TuckerOnNode::Tensor<ScalarType, Properties...> & dataTensor,
				  const std::optional<Tucker::SizeArray> & fixedCoreTensorRanks,
				  ScalarType tol)
{
  return [=](std::size_t mode, auto eigenValues) -> std::size_t
  {
    if (fixedCoreTensorRanks)
    {
      (void) eigenValues; // unused
      return (*fixedCoreTensorRanks)[mode];
    }
    else{
      (void) mode; // unused

      const auto rank = dataTensor.rank();
      const ScalarType norm = dataTensor.frobeniusNormSquared();
      const ScalarType threshold  = tol*tol*norm/rank;
      std::cout << "\tAutoST-HOSVD::Tensor Norm: "
		<< std::sqrt(norm)
		<< "...\n";
      std::cout << "\tAutoST-HOSVD::Relative Threshold: "
		<< threshold
		<< "...\n";
      auto eigVals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigenValues);
      return impl::count_eigvals_using_threshold(eigVals_h, threshold);
    }
  };
}

}//end namespace Tucker

#endif
