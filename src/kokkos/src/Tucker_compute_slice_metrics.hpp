#ifndef TUCKER_KOKKOS_COMPUTE_SLICE_METRICS_HPP_
#define TUCKER_KOKKOS_COMPUTE_SLICE_METRICS_HPP_

namespace Tucker {

template <class TensorType, class ScalarType>
auto compute_slice_metrics(const TensorType Y, const int mode, const int metrics)
{
  // If there are no slices, calling this function was a bad idea
  const int numSlices = Y.size(mode);
  if(numSlices <= 0) {
    std::ostringstream oss;
    oss << "Tucker::computeSliceMetrics(const Tensor<ScalarType>* Y, const int mode, const int metrics): "
        << "numSlices = " << numSlices << " <= 0";
    throw std::runtime_error(oss.str());
  }

  // Allocate memory for the resul
  Tucker::MetricData<ScalarType> result = Tucker::MetricData<ScalarType>(metrics, numSlices);

  // Initialize the result
  std::vector<ScalarType> delta;
  std::vector<int> nArray;

  if((metrics & MEAN) || (metrics & VARIANCE)) {
    delta = std::vector<ScalarType>(numSlices);
    nArray = std::vector<int>(numSlices);
  }

  for(int i=0; i<numSlices; i++) {
    if(metrics & MIN) {
      result.getMinData()[i] = std::numeric_limits<ScalarType>::max();
    }
    if(metrics & MAX) {
      result.getMaxData()[i] = std::numeric_limits<ScalarType>::lowest();
    }
    if(metrics & SUM) {
      result.getSumData()[i] = 0;
    }
    if((metrics & MEAN) || (metrics & VARIANCE)) {
      result.getMeanData()[i] = 0;
      nArray[i] = 0;
    }
    if(metrics & VARIANCE) {
      result.getVarianceData()[i] = 0;
    }
  }

  if(Y.getNumElements() == 0) {
    return result;
  }

  // Compute the result
  int ndims = Y.N();
  size_t numContig = Y.size().prod(0,mode-1,1); // Number of contiguous elements in a slice
  size_t numSetsContig = Y.size().prod(mode+1,ndims-1,1); // Number of sets of contiguous elements per slice
  size_t distBetweenSets = Y.size().prod(0,mode); // Distance between sets of contiguous elements

  const ScalarType* dataPtr;
  size_t i, c;
  Kokkos::parallel_for("computeResultData", numSlices, [&] (const int& slice) {
    dataPtr = Y.data().data() + slice*numContig;
    for(c=0; c<numSetsContig; c++)
    {
      for(i=0; i<numContig; i++)
      {
        if(metrics & MIN) {
          result.getMinData()[slice] = std::min(result.getMinData()[slice],dataPtr[i]);
        }
        if(metrics & MAX) {
          result.getMaxData()[slice] = std::max(result.getMaxData()[slice],dataPtr[i]);
        }
        if(metrics & SUM) {
          result.getSumData()[slice] += dataPtr[i];
        }
        if((metrics & MEAN) || (metrics & VARIANCE)) {
          delta[slice] = dataPtr[i] - result.getMeanData()[slice];
          nArray[slice]++;
          result.getMeanData()[slice] += (delta[slice]/nArray[slice]);
        }
        if(metrics & VARIANCE) {
          result.getVarianceData()[slice] +=
              (delta[slice]*(dataPtr[i]-result.getMeanData()[slice]));
        }
      } // end for(i=0; i<numContig; i++)
      dataPtr += distBetweenSets;
    } // end for(c=0; c<numSetsContig; c++)
  }); // end parallel_for

  if(metrics & VARIANCE) {
    size_t sizeOfSlice = numContig*numSetsContig;
    for(int i=0; i<numSlices; i++){
      result.getVarianceData()[i] /= (ScalarType)sizeOfSlice;
    }
  }

  return result;
}


}

#endif
