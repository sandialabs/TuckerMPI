#ifndef TUCKER_NORMALIZES_HPP_
#define TUCKER_NORMALIZES_HPP_

#include "Tucker_compute_slice_metrics.hpp"
#include "Tucker_transform_slices.hpp"
#include "TuckerOnNode_Tensor_io.hpp"

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
  const std::vector<Tucker::Metric> metrics{Tucker::Metric::MIN, Tucker::Metric::MAX};
  auto dataMetrics = TuckerOnNode::compute_slice_metrics(X, mode, metrics);

  int sizeOfModeDim = X.extent(mode);
  Kokkos::View<ScalarType*, MemorySpace> scales("scales", sizeOfModeDim);
  Kokkos::View<ScalarType*, MemorySpace> shifts("shifts", sizeOfModeDim);

  for(int i=0; i<sizeOfModeDim; i++) {
    scales(i) = dataMetrics.get(Tucker::Metric::MAX)[i] - dataMetrics.get(Tucker::Metric::MIN)[i];
    shifts(i) = -dataMetrics.get(Tucker::Metric::MIN)[i];
  }
  Tucker::transform_slices(X, mode, scales, shifts);

  if(scale_file) TuckerOnNode::write_scale_shift<ScalarType>(mode, sizeOfModeDim, scales, shifts, scale_file);
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

  if(scale_file) TuckerOnNode::write_scale_shift<ScalarType>(mode, sizeOfModeDim, scales, shifts, scale_file);
}

template <class ScalarType, class MemorySpace>
void normalize_tensor_standard_centering(TuckerOnNode::Tensor<ScalarType, MemorySpace> X,
		      int mode, ScalarType stdThresh, const char* scale_file=0)
{
  const std::vector<Tucker::Metric> metrics{Tucker::Metric::MEAN, Tucker::Metric::VARIANCE};
  auto dataMetrics = TuckerOnNode::compute_slice_metrics(X, mode, metrics);

  int sizeOfModeDim = X.extent(mode);
  Kokkos::View<ScalarType*, MemorySpace> scales("scales", sizeOfModeDim);
  Kokkos::View<ScalarType*, MemorySpace> shifts("shifts", sizeOfModeDim);

  for(int i=0; i<sizeOfModeDim; i++) {
    scales(i) = sqrt(dataMetrics.get(Tucker::Metric::VARIANCE)[i]);
    shifts(i) = -dataMetrics.get(Tucker::Metric::MEAN)[i];

    if(scales(i) < stdThresh) {
      scales(i) = 1;
    }
  }
  Tucker::transform_slices(X, mode, scales, shifts);

  if(scale_file) TuckerOnNode::write_scale_shift<ScalarType>(mode, sizeOfModeDim, scales, shifts, scale_file);
}

}//end namespace Tucker

#endif
