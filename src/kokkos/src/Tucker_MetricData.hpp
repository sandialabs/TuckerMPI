#ifndef TUCKER_KOKKOS_METRIC_DATA_HPP_
#define TUCKER_KOKKOS_METRIC_DATA_HPP_

#include <exception>

namespace Tucker {

enum Metric {
  MIN = 1, ///< Minimum value
  MAX = 2, ///< Maximum value
  SUM = 4, ///< Sum of all values
  NORM1 = 8, ///< 1-norm
  NORM2 = 16, ///< 2-norm
  MEAN = 32, ///< Mean
  VARIANCE = 64 ///< Variance
};

// Class that stores all metrics
template <class ScalarType>
class MetricData
{
public:
  // Constructor
  MetricData(const int metrics, const int dimension) : dimension_(dimension)
  {
    assert(dimension > 0);

    if(metrics & MIN)
      minData_ = std::vector<ScalarType>(dimension);

    if(metrics & MAX)
      maxData_ = std::vector<ScalarType>(dimension);

    if(metrics & SUM)
      sumData_ = std::vector<ScalarType>(dimension);

    if(metrics & NORM1)
      norm1Data_ = std::vector<ScalarType>(dimension);

    if(metrics & NORM2)
      norm2Data_ = std::vector<ScalarType>(dimension);

    if(metrics & MEAN)
      meanData_ = std::vector<ScalarType>(dimension);

    if(metrics & VARIANCE)
      varianceData_ = std::vector<ScalarType>(dimension);
  }

  // Returns the min-data array
  ScalarType* getMinData()
  {
    if(minData_.size() != 0) return minData_.data();
    throw std::runtime_error("min data was never allocated!");
  }

  // Returns the max-data array
  ScalarType* getMaxData()
  {
    if(maxData_.size() != 0) return maxData_.data();
    throw std::runtime_error("max data was never allocated!");
  }

  // Returns the sum-data array
  ScalarType* getSumData()
  {
    if(sumData_.size() != 0) return sumData_.data();
    throw std::runtime_error("sum data was never allocated!");
  }

  // Returns the norm1-data array
  ScalarType* getNorm1Data()
  {
    if(norm1Data_.size() != 0) return norm1Data_.data();
    throw std::runtime_error("norm1 data was never allocated!");
  }

  // Returns the norm2-data array
  ScalarType* getNorm2Data()
  {
    if(norm2Data_.size() != 0) return norm2Data_.data();
    throw std::runtime_error("norm2 data was never allocated!");
  }

  // Returns the mean-data array
  ScalarType* getMeanData()
  {
    if(meanData_.size() != 0) return meanData_.data();
    throw std::runtime_error("mean data was never allocated!");
  }

  // Returns the variance array
  ScalarType* getVarianceData()
  {
    if(varianceData_.size() != 0) return varianceData_.data();
    throw std::runtime_error("variance data was never allocated!");
  }

private:
  std::vector<ScalarType> minData_; ///< Minimum value array
  std::vector<ScalarType> maxData_; ///< Maximum value array
  std::vector<ScalarType> sumData_; ///< Sum of all values array
  std::vector<ScalarType> norm1Data_; ///< 1-norm array
  std::vector<ScalarType> norm2Data_; ///< 2-norm array
  std::vector<ScalarType> meanData_; ///< Mean array
  std::vector<ScalarType> varianceData_; ///< Variance array
  int dimension_;
};

// Explicit instantiations to build static library for both single and double precision
template class MetricData<float>;
template class MetricData<double>;

}

#endif