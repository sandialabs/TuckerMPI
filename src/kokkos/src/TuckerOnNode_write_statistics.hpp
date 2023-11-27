#ifndef TUCKERONNODE_WRITE_STATISTICS_HPP_
#define TUCKERONNODE_WRITE_STATISTICS_HPP_

#include "Tucker_fwd.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include <fstream>
#include <iostream>

namespace TuckerOnNode{

template<class ScalarType, class MemSpace>
void write_statistics(TuckerOnNode::MetricData<ScalarType, MemSpace> metricData,
		      const std::string & statsFile,
		      const ScalarType stdThresh)
{
  auto metricData_h = Tucker::create_mirror(metricData);
  Tucker::deep_copy(metricData_h, metricData);

  std::cout << "Writing file " << statsFile << std::endl;
  std::ofstream statStream(statsFile);
  statStream << std::setw(5) << "Mode"
      << std::setw(13) << "Mean"
      << std::setw(13) << "Stdev"
      << std::setw(13) << "Min"
      << std::setw(13) << "Max"
      << std::endl;

  auto view_max = metricData_h.get(Tucker::Metric::MAX);
  auto view_min = metricData_h.get(Tucker::Metric::MIN);
  auto view_mean = metricData_h.get(Tucker::Metric::MEAN);
  auto view_variance = metricData_h.get(Tucker::Metric::VARIANCE);
  const std::size_t count = view_max.extent(0);

  for(std::size_t i=0; i<count; i++) {
    double stdev = sqrt(view_variance(i));

    if(stdev < stdThresh) {
      std::cout << "Slice " << i
		<< " is below the cutoff. True value is: "
		<< stdev << std::endl;
      stdev = 1.;
    }
    statStream << std::setw(5) << i
	       << std::setw(13) << view_mean(i)
	       << std::setw(13) << stdev
	       << std::setw(13) << view_min(i)
	       << std::setw(13) << view_max(i)
	       << std::endl;
  }
  statStream.close();
}

}//end namespace Tucker

#endif  // TUCKERONNODE_WRITE_STATISTICS_HPP_
