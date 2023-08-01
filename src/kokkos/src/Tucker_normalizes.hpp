#ifndef TUCKER_NORMALIZES_HPP_
#define TUCKER_NORMALIZES_HPP_

#include "Tucker_fwd.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_compute_slice_metrics.hpp"

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

  //int sizeOfModeDim = X.size(mode);

  /*
  scalar_t* scales = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  scalar_t* shifts = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scalar_t scaleval = std::max(std::abs(metrics->getMinData()[i]),
        std::abs(metrics->getMaxData()[i]));
    scales[i] = scaleval;
    shifts[i] = 0;
  }
  transformSlices(Y,mode,scales,shifts);
  if(scale_file) writeScaleShift(mode,sizeOfModeDim,scales,shifts,scale_file);
  */
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
