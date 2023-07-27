#ifndef TUCKER_WRITE_STATISTICS_HPP_
#define TUCKER_WRITE_STATISTICS_HPP_

#include <fstream>

namespace Tucker{

template <class MetricsType, class ScalarType>
void write_statistics(const MetricsType metrics,
		      const std::string statsFile,
		      const ScalarType stdThresh)
{

  // std::cout << "Writing file " << statsFile << std::endl;
  // std::ofstream statStream(statsFile);
  // statStream << std::setw(5) << "Mode"
  //     << std::setw(13) << "Mean"
  //     << std::setw(13) << "Stdev"
  //     << std::setw(13) << "Min"
  //     << std::setw(13) << "Max"
  //     << std::endl;

  // for(int i=0; i<dataTensor.extent(scaleMode); i++) {
  //   double stdev = sqrt(metrics.getVarianceData()[i]);
  //   if(stdev < stdThresh) {
  //     std::cout << "Slice " << i
  //         << " is below the cutoff. True value is: "
  //         << stdev << std::endl;
  //     stdev = 1;
  //   }
  //   statStream << std::setw(5) << i
  //       << std::setw(13) << metrics.getMeanData()[i]
  //       << std::setw(13) << stdev
  //       << std::setw(13) << metrics.getMinData()[i]
  //       << std::setw(13) << metrics.getMaxData()[i] << std::endl;
  // }
  // statStream.close();
}

}//end namespace Tucker

#endif
