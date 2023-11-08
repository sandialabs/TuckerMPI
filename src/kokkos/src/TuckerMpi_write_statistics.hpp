#ifndef TUCKERMPI_WRITE_STATISTICS_HPP_
#define TUCKERMPI_WRITE_STATISTICS_HPP_

#include "Tucker_fwd.hpp"
#include "TuckerMpi_Distribution.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include "./impl/TuckerMpi_MPIWrapper.hpp"
#include <fstream>

namespace TuckerMpi{

template<class ScalarType, class MemSpace>
void write_statistics(const int mpiRank,
                      const int tensorRank,
                      const int scaleMode,
                      const Distribution & distribution,
                      TuckerOnNode::MetricData<ScalarType, MemSpace> metricData,
                      const std::string & outputFile,
                      const ScalarType stdThresh)
{
  // for the code below to work, we need the metricsData to be accessible on host
  auto metricData_h = Tucker::create_mirror(metricData);
  Tucker::deep_copy(metricData_h, metricData);

  // metricData must have MIN, MAX, MEAN, VARIANCE because the code below
  // is hardwired for writing to file those stats
  const std::vector<Tucker::Metric> metricIDsNeeded{Tucker::Metric::MIN,
                                                    Tucker::Metric::MAX,
                                                    Tucker::Metric::MEAN,
                                                    Tucker::Metric::VARIANCE};
  for (auto & it : metricIDsNeeded){
    if (!metricData_h.contains(it)){
      throw std::runtime_error("TuckerMpi::write_statistics: metricData MUST contain MIN, MAX, MEAN, VARIANCE");
    }
  }

  //
  // Determine whether I need to communicate with rank 0
  //
  std::vector<int> myCoordinates(tensorRank);
  std::vector<int> zeroCoordinates(tensorRank);
  auto grid = distribution.getProcessorGrid();
  grid.getCoordinates(myCoordinates);
  grid.getCoordinates(zeroCoordinates, 0);
  bool needToSendToZero = true;
  for(int i=0; i<tensorRank; i++) {
    if(i == scaleMode) continue;

    if(myCoordinates[i] != zeroCoordinates[i]) {
      needToSendToZero = false;
      break;
    }
  }

  const auto map = distribution.getMap(scaleMode);
  const MPI_Comm& rowComm = grid.getColComm(scaleMode);
  const std::size_t metricsCount = metricData_h.numMetricsStored();
  if(needToSendToZero)
  {
    const std::size_t globalNumEntries = map->getGlobalNumEntries();
    std::unordered_map<Tucker::Metric, int> metricNameToColIndex;
    Kokkos::View<ScalarType**, Kokkos::LayoutLeft, Kokkos::HostSpace> M(
      "M", globalNumEntries, metricsCount);
    for (std::size_t i=0; i<metricIDsNeeded.size(); ++i){
      const auto metricID = metricIDsNeeded[i];
      metricNameToColIndex[metricID] = i;
      auto view = metricData_h.get(metricID);
      std::vector<ScalarType> tmp(view.extent(0));
      Tucker::impl::copy_view_to_stdvec(view, tmp);
      auto dest_col = Kokkos::subview(M, Kokkos::ALL, i);
      TuckerMpi::MPI_Gatherv_(tmp.data(),
                              map->getLocalNumEntries(),
                              dest_col.data(), // can do this because of LayoutLeft
                              (int*)map->getNumElementsPerProc().data(),
                              (int*)map->getOffsets().data(),
                              0, rowComm);
    }

    //
    // write stats to file
    //
    if(mpiRank == 0) {
      std::cout << "Writing file " << outputFile << std::endl;
      std::ofstream statStream(outputFile);
      statStream << std::setw(5)  << "Mode"
                 << std::setw(13) << "Mean"
                 << std::setw(13) << "Stdev"
                 << std::setw(13) << "Min"
                 << std::setw(13) << "Max"
                 << std::endl;

      const int min_colInd = metricNameToColIndex[Tucker::Metric::MIN];
      const int max_colInd = metricNameToColIndex[Tucker::Metric::MAX];
      const int mean_colInd = metricNameToColIndex[Tucker::Metric::MEAN];
      const int var_colInd  = metricNameToColIndex[Tucker::Metric::VARIANCE];

      for(int i=0; i<(int)M.extent(0); i++){
        double stdev = std::sqrt(M(i, var_colInd));
        if(stdev < stdThresh) {
          std::cout << "Slice " << i << " is below the cutoff. True value is: " << stdev << std::endl;
          stdev = 1;
        }

        statStream << std::setw(5) << i
                   << std::setw(13) << M(i, mean_colInd)
                   << std::setw(13) << stdev
                   << std::setw(13) << M(i, min_colInd)
                   << std::setw(13) << M(i, max_colInd)
                   << std::endl;
      }

      statStream.close();
    }//end if mpiRank==0
  }
}

}//end namespace Tucker

#endif  // TUCKERMPI_WRITE_STATISTICS_HPP_
