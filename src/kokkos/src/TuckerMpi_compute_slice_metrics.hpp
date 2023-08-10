
#ifndef TUCKER_MPI_COMPUTE_SLICE_METRICS_HPP_
#define TUCKER_MPI_COMPUTE_SLICE_METRICS_HPP_

#include "TuckerOnNode_compute_slice_metrics.hpp"
#include "./impl/TuckerMpi_prod_impl.hpp"
#include "./impl/TuckerMpi_MPIWrapper.hpp"

namespace TuckerMpi{

template <std::size_t n, class ScalarType, class ...Properties>
auto compute_slice_metrics(const int mpiRank,
			   Tensor<ScalarType, Properties...> tensor,
			   const int mode,
			   const std::array<Tucker::Metric, n> & metrics)
{
  using tensor_type = Tensor<ScalarType, Properties...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;
  using tensor_layout = typename tensor_type::traits::onnode_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;

  // constraints
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point_v<tensor_value_type>,
		   "TuckerMpi::compute_slice_metrics: supports tensors with LayoutLeft" \
		   "and floating point scalar");

  // preconditions
  if(tensor.localExtent(mode) <= 0) {
    std::ostringstream oss;
    oss << "TuckerMpi::compute_slice_metrics: "
        << "for mode = " << mode
	<< " we have tensor.localExtent(mode) = " << tensor.localExtent(mode) << " <= 0";
    throw std::runtime_error(oss.str());
  }

  if(mode < 0) {
    throw std::runtime_error("mode must be non-negative");
  }

  // Compute the local result
  auto localTensor = tensor.localTensor();
  auto result = TuckerOnNode::compute_slice_metrics(localTensor, mode, metrics);
  auto result_h = ::Tucker::create_mirror(result);
  ::Tucker::deep_copy(result_h, result);

  // Get the row communicator
  const MPI_Comm& comm = tensor.getDistribution().getProcessorGrid().getRowComm(mode, false);
  int nprocs;
  MPI_Comm_size(comm, &nprocs);

  if(nprocs > 1)
  {
    // Compute the global result
    const int numSlices = tensor.localExtent(mode);
    std::vector<tensor_value_type> sendBuf(numSlices);
    std::vector<tensor_value_type> recvBuf(numSlices);

    if (result_h.contains(Tucker::Metric::MIN)) {
      auto view_h = result_h.get(Tucker::Metric::MIN);
      for(int i=0; i<numSlices; i++) sendBuf[i] = view_h(i);
      MPI_Allreduce_(sendBuf.data(), recvBuf.data(), numSlices, MPI_MIN, comm);
      for(int i=0; i<numSlices; i++) view_h(i) = recvBuf[i];
    }

    if (result_h.contains(Tucker::Metric::MAX)) {
      auto view_h = result_h.get(Tucker::Metric::MAX);
      for(int i=0; i<numSlices; i++) sendBuf[i] = view_h(i);
      MPI_Allreduce_(sendBuf.data(), recvBuf.data(), numSlices, MPI_MAX, comm);
      for(int i=0; i<numSlices; i++) view_h(i) = recvBuf[i];
    }

    if (result_h.contains(Tucker::Metric::SUM)) {
      auto view_h = result_h.get(Tucker::Metric::SUM);
      for(int i=0; i<numSlices; i++) sendBuf[i] = view_h(i);
      MPI_Allreduce_(sendBuf.data(), recvBuf.data(), numSlices, MPI_SUM, comm);
      for(int i=0; i<numSlices; i++) view_h(i) = recvBuf[i];
    }

    // If X is partitioned into X_A and X_B: mean_X = (n_A mean_A + n_B mean_B) / (n_A + n_B)
    if ( result_h.contains(Tucker::Metric::MEAN) ||
	 result_h.contains(Tucker::Metric::VARIANCE) )
    {
      // Compute the size of my local slice
      const int ndims = tensor.rank();
      auto localSize = tensor.localDimensionsOnHost();
      std::size_t localSliceSize = impl::prod(localSize, 0,mode-1,1) * impl::prod(localSize, mode+1, ndims-1,1);

      // Compute the size of the global slice
      auto globalSize = tensor.globalDimensionsOnHost();
      std::size_t globalSliceSize = impl::prod(globalSize, 0, mode-1, 1) * impl::prod(globalSize, mode+1, ndims-1, 1);
      auto mean_view_h = result_h.get(Tucker::Metric::MEAN);
      for(int i=0; i<numSlices; i++) {
        sendBuf[i] = mean_view_h(i) * (ScalarType)localSliceSize;
      }
      MPI_Allreduce_(sendBuf.data(), recvBuf.data(), numSlices, MPI_SUM, comm);

      std::vector<ScalarType> meanDiff;
      if(result_h.contains(Tucker::Metric::VARIANCE)) {
        meanDiff.resize(numSlices);
        for(int i=0; i<numSlices; i++) {
          meanDiff[i] = mean_view_h(i) - recvBuf[i] / (ScalarType)globalSliceSize;
        }
      }
      for(int i=0; i<numSlices; i++) {
        mean_view_h(i) = recvBuf[i] / (ScalarType)globalSliceSize;
      }

      if(result_h.contains(Tucker::Metric::VARIANCE)) {
	auto var_view_h = result_h.get(Tucker::Metric::VARIANCE);
        for(int i=0; i<numSlices; i++) {
          // Source of this equation:
          // http://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
          sendBuf[i] = (ScalarType)localSliceSize* var_view_h(i) +
	    (ScalarType)localSliceSize*meanDiff[i]*meanDiff[i];
        }

        MPI_Allreduce_(sendBuf.data(), recvBuf.data(), numSlices, MPI_SUM, comm);
        for(int i=0; i<numSlices; i++) {
	  var_view_h(i) = recvBuf[i] / (ScalarType)globalSliceSize;
        }
      } // end if(metrics & Tucker::VARIANCE)

    } // end if((metrics & Tucker::MEAN) || (metrics & Tucker::VARIANCE))

  } // end if(nprocs > 1)

  ::Tucker::deep_copy(result, result_h);
  return result;
}

}
#endif
