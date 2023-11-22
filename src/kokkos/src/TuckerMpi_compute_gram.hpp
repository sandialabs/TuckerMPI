#ifndef TUCKERMPI_COMPUTE_GRAM_HPP_
#define TUCKERMPI_COMPUTE_GRAM_HPP_

#include "TuckerMpi_Tensor.hpp"
#include "./impl/TuckerMpi_newgram_impl.hpp"
#include "Tucker_Timer.hpp"

namespace TuckerMpi{

template<class ScalarType, class ...Properties>
[[nodiscard]] auto compute_gram(Tensor<ScalarType, Properties...> tensor,
                                const std::size_t n,
                                Tucker::Timer* matmul_timer = nullptr,
                                Tucker::Timer* pack_timer = nullptr,
                                Tucker::Timer* alltoall_timer = nullptr,
                                Tucker::Timer* unpack_timer = nullptr,
                                Tucker::Timer* allreduce_timer = nullptr)
{

  using tensor_type       = Tensor<ScalarType, Properties...>;
  using memory_space      = typename tensor_type::traits::memory_space;
  using onnode_layout     = typename tensor_type::traits::onnode_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;
  using gram_type         = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space>;

  // constraints
  static_assert(   std::is_same_v<onnode_layout, Kokkos::LayoutLeft>
    && std::is_same_v<std::remove_cv_t<tensor_value_type>, double>,
                   "TuckerMpi::compute_gram: supports tensors with LayoutLeft" \
                   "and double scalar type");

  //
  // compute local gram
  gram_type localGram;

  const MPI_Comm& comm = tensor.getDistribution().getProcessorGrid().getColComm(n);
  int numProcs;
  MPI_Comm_size(comm, &numProcs);
  if(numProcs > 1){
    impl::local_gram_after_data_redistribution(
      tensor, n, localGram, matmul_timer, pack_timer, alltoall_timer,
      unpack_timer);
  }
  else{
    impl::local_gram_without_data_redistribution(
      tensor, n, localGram, matmul_timer);
  }

  //
  // now do reduction across mpi ranks
  const std::size_t nrows = localGram.extent(0);
  const std::size_t ncols = localGram.extent(1);
  const std::size_t count = nrows*ncols;

  // FIXME: why we must use MPI_COMM_WORLD or all tests fails?
  Kokkos::fence(); // Need fence to ensure local gram is finished before reduce
  if (allreduce_timer) allreduce_timer->start();
  MPI_Allreduce_(localGram.data(), count, MPI_SUM, MPI_COMM_WORLD);
  if (allreduce_timer) allreduce_timer->stop();

  return localGram;
}

}
#endif  // TUCKERMPI_COMPUTE_GRAM_HPP_
