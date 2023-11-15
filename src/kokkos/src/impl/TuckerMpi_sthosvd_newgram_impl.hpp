#ifndef IMPL_TUCKERMPI_STHOSVD_NEWGRAM_IMPL_HPP_
#define IMPL_TUCKERMPI_STHOSVD_NEWGRAM_IMPL_HPP_

#include "TuckerMpi_compute_gram.hpp"
#include "Tucker_TuckerTensor.hpp"
#include "Tucker_Timer.hpp"

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

  Tucker::Timer sthosvd_timer;
  sthosvd_timer.start();

  // ---------------------
  // prepare
  // ---------------------
  int mpiRank;
  int nprocs;
  MPI_Comm_rank(myComm, &mpiRank);
  MPI_Comm_size(myComm, &nprocs);

  // Compute the nnz of the largest tensor piece being stored by any process
  std::size_t max_lcl_nnz_x = 1;
  for(int i=0; i<X.rank(); i++) {
    max_lcl_nnz_x *= X.getDistribution().getMap(i)->getMaxNumEntries();
  }
  MPI_Barrier(myComm);

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
    const int mode = modeOrder.empty() ? n : modeOrder[n];

    if(mpiRank == 0) {
      std::cout << "\n---------------------------------------------\n";
      std::cout << "--- AutoST-HOSVD::Starting Mode(" << n << ") --- \n";
      std::cout << "---------------------------------------------\n";
    }
    Tucker::Timer sthosvd_iter_timer;
    sthosvd_iter_timer.start();

    /*
     * GRAM
     */
    Tucker::Timer gram_timer, matmul_timer, pack_timer, alltoall_timer, unpack_timer, allreduce_timer;
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Starting Gram(" << mode << ") \n";
    }
    gram_timer.start();
    auto S = ::TuckerMpi::compute_gram(
      Y, mode, &matmul_timer, &pack_timer, &alltoall_timer, &unpack_timer,
      &allreduce_timer);
    gram_timer.stop();
    if(mpiRank == 0) {
      std::cout << "    Gram(" << mode << ")::Local Matmul time: "
                << matmul_timer.duration() << "s\n";
      std::cout << "    Gram(" << mode << ")::Pack time: "
                << pack_timer.duration() << "s\n";
      std::cout << "    Gram(" << mode << ")::All-to-all time: "
                << alltoall_timer.duration() << "s\n";
      std::cout << "    Gram(" << mode << ")::Unpack time: "
                << unpack_timer.duration() << "s\n";
      std::cout << "    Gram(" << mode << ")::All-reduce time: "
                << allreduce_timer.duration() << "s\n";
      std::cout << "  AutoST-HOSVD::Gram(" << mode << ") time: "
                << gram_timer.duration() << "s\n";
    }

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
    Tucker::Timer eigen_timer;
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Starting Evecs(" << mode << ")...\n";
    }
    eigen_timer.start();
    auto currEigvals = Tucker::impl::compute_and_sort_descending_eigvals_and_eigvecs_inplace(S, flipSign);
    appendEigenvaluesAndUpdateSliceInfo(mode, eigvals, currEigvals, perModeSlicingInfo_eigvals(mode));
    eigen_timer.stop();
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Evecs(" << mode << ") time: "
                << eigen_timer.duration() << "s\n";
    }

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
    Tucker::Timer truncate_timer;
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Starting Truncate(" << mode << ")...\n";
    }
    truncate_timer.start();
    const std::size_t numEvecs = truncator(mode, currEigvals);
    auto currEigVecs = Kokkos::subview(S, Kokkos::ALL, std::pair<std::size_t,std::size_t>{0, numEvecs});
    appendFactorsAndUpdateSliceInfo(mode, factors, currEigVecs, perModeSlicingInfo_factors(mode));
    truncate_timer.stop();
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Truncate(" << mode << ") time: "
                << truncate_timer.duration() << "s\n";
    }

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
    Tucker::Timer ttm_timer;
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Starting TTM(" << mode << ")...\n";
    }
    ttm_timer.start();
    tensor_type temp = ::TuckerMpi::ttm(Y, mode, currEigVecs, true, max_lcl_nnz_x);
    ttm_timer.stop();
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::TTM(" << mode << ") time: "
                << ttm_timer.duration() << "s\n";
    }

    // need to do = {} first, otherwise Y=temp throws because Y = temp
    // is assigning tensors with different distributions
    Y = {};
    Y = temp;
    MPI_Barrier(myComm);

    if(mpiRank == 0) {
      const std::size_t local_nnz = Y.localSize();
      const std::size_t global_nnz = Y.globalSize();

      std::cout << "Local tensor size after STHOSVD iteration  " << mode << ": ";
      Tucker::write_view_to_stream_singleline(std::cout, Y.localDimensionsOnHost());
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(ScalarType));

      std::cout << "Global tensor size after STHOSVD iteration " << mode << ": ";
      Tucker::write_view_to_stream_singleline(std::cout, Y.globalDimensionsOnHost());
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(ScalarType));
    }

    sthosvd_iter_timer.stop();
    if (mpiRank == 0){
      std::cout << "  AutoST-HOSVD(" << mode << ") time: "
                << sthosvd_iter_timer.duration() << "s\n";
    }

  }//end loop

  sthosvd_timer.stop();
  if (mpiRank == 0){
    std::cout << "STHOSVD time: " << sthosvd_timer.duration()  << "s\n";
  }

  return std::pair( tucker_tensor_type(Y, factors, perModeSlicingInfo_factors),
                    gram_eigvals_type(eigvals, perModeSlicingInfo_eigvals) );
}

}}
#endif  // IMPL_TUCKERMPI_STHOSVD_NEWGRAM_IMPL_HPP_
