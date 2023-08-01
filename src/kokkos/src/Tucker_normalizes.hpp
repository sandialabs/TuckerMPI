#ifndef TUCKER_NORMALIZES_HPP_
#define TUCKER_NORMALIZES_HPP_

#include "Tucker_compute_slice_metrics.hpp"
#include "Tucker_transform_slices.hpp"

/** \brief Normalize each slice of the tensor so its data lies in the range [0,1]
 *
 * \param Y The tensor whose slices are being normalized
 * \param mode The mode which determines the slices
 */

namespace Tucker{

template <class ScalarType, class MemorySpace>
void normalize_tensor_min_max(TuckerOnNode::Tensor<ScalarType, MemorySpace> X,
		      int mode, const char* scale_file=0)
{
  //FIXME: TODO
  std::cout << "TODO: normalizeTensorMinMax" << std::endl;
}

template <class ScalarType, class MemorySpace>
void normalize_tensor_max(TuckerOnNode::Tensor<ScalarType, MemorySpace> X,
		      int mode, const char* scale_file=0)
{
  const std::vector<Tucker::Metric> metrics{Tucker::Metric::MIN, Tucker::Metric::MAX};
  auto dataMetrics = TuckerOnNode::compute_slice_metrics(X, mode, metrics);

  int sizeOfModeDim = X.extent(mode);
  Kokkos::View<ScalarType*, MemorySpace> scales("scales", sizeOfModeDim);
  Kokkos::View<ScalarType*, MemorySpace> shifts("shifts", sizeOfModeDim);

  for(int i=0; i<sizeOfModeDim; i++) {
    ScalarType scaleval = std::max(
      std::abs(dataMetrics.get(Tucker::Metric::MIN)[i]),
      std::abs(dataMetrics.get(Tucker::Metric::MAX)[i])
    );
    scales(i) = scaleval;
    shifts(i) = 0;
  }
  Tucker::transform_slices(X, mode, scales, shifts);
  // if(scale_file) write_scale_shift(mode, sizeOfModeDim, scales, shifts, scale_file);
}

template <class ScalarType, class MemorySpace>
void normalize_tensor_standard_centering(TuckerOnNode::Tensor<ScalarType, MemorySpace> X,
		      int mode, ScalarType stdThresh, const char* scale_file=0)
{
  //FIXME: TODO
  std::cout << "TODO: normalizeTensorStandardCentering" << std::endl;
}

}//end namespace Tucker

#endif
