#ifndef IMPL_TUCKERMPI_STHOSVD_NEWGRAM_IMPL_HPP_
#define IMPL_TUCKERMPI_STHOSVD_NEWGRAM_IMPL_HPP_

#include "TuckerMpi_compute_gram.hpp"
#include "Tucker_TuckerTensor.hpp"
#include <chrono>

namespace TuckerMpi{
namespace impl{

template <class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto sthosvd_newgram(Tensor<ScalarType, Properties...> X,
				   TruncatorType && truncator,
				   const std::vector<int> & modeOrder,
				   bool flipSign)
{
  using tensor_type         = Tensor<ScalarType, Properties...>;
  using memory_space        = typename tensor_type::traits::memory_space;
  using tucker_tensor_type  = Tucker::TuckerTensor<tensor_type>;
  using gram_eigvals_type   = TuckerOnNode::TensorGramEigenvalues<ScalarType, memory_space>;
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  MPI_Comm myComm = MPI_COMM_WORLD;

  // ---------------------
  // prepare
  // ---------------------
  int mpiRank;
  int nprocs;
  MPI_Comm_rank(myComm, &mpiRank);
  MPI_Comm_size(myComm, &nprocs);

  // Compute the nnz of the largest tensor piece being stored by any process
  size_t max_lcl_nnz_x = 1;
  for(int i=0; i<X.rank(); i++) {
    max_lcl_nnz_x *= X.getDistribution().getMap(i,false)->getMaxNumEntries();
  }
  MPI_Barrier(myComm);

  auto outerStart = std::chrono::high_resolution_clock::now();

  // ---------------------
  // core loop
  // ---------------------
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> eigvals;
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> factors;
  slicing_info_view_t perModeSlicingInfo_factors("pmsi_factors", X.rank());
  slicing_info_view_t perModeSlicingInfo_eigvals("pmsi_eigvals", X.rank());

  tensor_type Y = X;
  for (std::size_t n=0; n<(std::size_t)X.rank(); n++)
  {
    auto innerStart = std::chrono::high_resolution_clock::now();
    const int mode = modeOrder.empty() ? n : modeOrder[n];

    if(mpiRank == 0) {
      std::cout << "\n---------------------------------------------\n";
      std::cout << "--- AutoST-HOSVD::Starting Mode(" << n << ") --- \n";
      std::cout << "---------------------------------------------\n";
    }

    /*
     * GRAM
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Gram(" << mode << ") \n";
    }
    auto S = ::TuckerMpi::compute_gram(Y, mode);
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank == 0){
      std::cout << "\n";
      Tucker::write_view_to_stream(std::cout, S);
      std::cout << "\n";
    }
#endif

    /*
     * eigenvalues and eigenvectors
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Eigen{vals,vecs}(" << mode << ")...\n";
    }
    auto currEigvals = Tucker::impl::compute_and_sort_descending_eigvals_and_eigvecs_inplace(S, flipSign);
    appendEigenvaluesAndUpdateSliceInfo(mode, eigvals, currEigvals, perModeSlicingInfo_eigvals(mode));
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank == 0){
      std::cout << "\n";
      Tucker::write_view_to_stream(std::cout, currEigvals);
      std::cout << "\n";
    }
#endif

    /*
     * Truncation
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Truncating\n";
    }
    const std::size_t numEvecs = truncator(mode, currEigvals);
    auto currEigVecs = Kokkos::subview(S, Kokkos::ALL, std::pair<std::size_t,std::size_t>{0, numEvecs});
    appendFactorsAndUpdateSliceInfo(mode, factors, currEigVecs, perModeSlicingInfo_factors(mode));
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank ==0){
      std::cout << "\n";
      Tucker::write_view_to_stream(std::cout, currEigVecs);
      std::cout << "\n";
    }
#endif

    /*
     * TTM
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Starting TTM(" << mode << ")...\n";
    }
    auto ttmStart = std::chrono::high_resolution_clock::now();

    tensor_type temp = ::TuckerMpi::ttm(Y, mode, currEigVecs, true, max_lcl_nnz_x);

    auto ttmStop = std::chrono::high_resolution_clock::now();
    auto ttmDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(ttmStop - ttmStart);
    const double ttmDurationDbl = ttmDuration.count() * 1e-9;
    double ttmGlobalMean = {};
    MPI_Reduce(&ttmDurationDbl, &ttmGlobalMean, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
    ttmGlobalMean /= (double) nprocs;
    if (mpiRank == 0){
      std::cout << "  TTM time (sec): " << std::setprecision(6) << ttmGlobalMean << std::endl;
    }

    // need to do = {} first, otherwise Y=temp throws because Y = temp
    // is assigning tensors with different distributions
    Y = {};
    Y = temp;
    MPI_Barrier(myComm);

    if(mpiRank == 0) {
      const size_t local_nnz = Y.localSize();
      const size_t global_nnz = Y.globalSize();

      std::cout << "Local tensor size after STHOSVD iteration  " << mode << ": ";
      Tucker::write_view_to_stream_singleline(std::cout, Y.localDimensionsOnHost());
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(ScalarType));

      std::cout << "Global tensor size after STHOSVD iteration " << mode << ": ";
      Tucker::write_view_to_stream_singleline(std::cout, Y.globalDimensionsOnHost());
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(ScalarType));
    }

    auto innerStop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(innerStop - innerStart);
    const double innerDurationDbl = duration.count() * 1e-9;
    double innerGlobalMean = {};
    MPI_Reduce(&innerDurationDbl, &innerGlobalMean, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
    innerGlobalMean /= (double) nprocs;
    if (mpiRank == 0){
      std::cout << "  STHOSVD time (sec): " << std::setprecision(6) << innerGlobalMean << std::endl;
    }

  }//end loop

  auto outerStop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(outerStop - outerStart);
  const double outerDurationDbl = duration.count() * 1e-9;
  double outerGlobalMean = {};
  MPI_Reduce(&outerDurationDbl, &outerGlobalMean, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
  outerGlobalMean /= (double) nprocs;
  if (mpiRank == 0){
    std::cout << "STHOSVD time (sec): " << std::setprecision(6) << outerGlobalMean << std::endl;
  }

  return std::pair( tucker_tensor_type(Y, factors, perModeSlicingInfo_factors),
		    gram_eigvals_type(eigvals, perModeSlicingInfo_eigvals) );
}

}}
#endif  // IMPL_TUCKERMPI_STHOSVD_NEWGRAM_IMPL_HPP_
