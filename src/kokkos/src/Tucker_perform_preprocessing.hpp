#ifndef TUCKER_PERFORM_PREPROCESSING_HPP_
#define TUCKER_PERFORM_PREPROCESSING_HPP_

#include "TuckerOnNode_Tensor_io.hpp"
#include "Tucker_normalizes.hpp"

namespace Tucker{

template <class ScalarType, class MemorySpace>
void perform_preprocessing(const TuckerOnNode::Tensor<ScalarType, MemorySpace> X,
		      const std::string sthosvdDir,
          const std::string sthosvdFn,
          const std::string scalingType,
          const int scaleMode,
          const ScalarType stdThresh,
          const bool writePreprocessed,
          const std::string preFnsFile)
{
  std::string scale_file = sthosvdDir + "/" + sthosvdFn + "_scale.txt";
  if(scalingType == "Max") {
    std::cout << "Normalizing the tensor by maximum entry - mode " << scaleMode << std::endl;
    Tucker::normalize_tensor_max(X, scaleMode, scale_file.c_str());
  }
  else if(scalingType == "MinMax") {
    std::cout << "Normalizing the tensor using minmax scaling - mode " << scaleMode << std::endl;
    Tucker::normalize_tensor_min_max(X, scaleMode, scale_file.c_str());
  }
  else if(scalingType == "StandardCentering") {
    std::cout << "Normalizing the tensor using standard centering - mode " << scaleMode << std::endl;
    Tucker::normalize_tensor_standard_centering(X, scaleMode, stdThresh, scale_file.c_str());
  }
  else if(scalingType == "None") {
    std::cout << "Not scaling the tensor\n";
  }
  else {
    std::cerr << "Error: invalid scaling type: " << scalingType << std::endl;
  }

  if(writePreprocessed) {
    TuckerOnNode::write_tensor_binary(X, preFnsFile.c_str());
  }
}

}//end namespace Tucker

#endif