#ifndef TUCKERKOKKOS_MPI_COMP_GRAM_HPP_
#define TUCKERKOKKOS_MPI_COMP_GRAM_HPP_

#include "TuckerMpi_Tensor.hpp"
#include "./impl/TuckerMpi_newgram_impl.hpp"

namespace TuckerMpi{

template<class ScalarType, class ...Properties>
auto compute_gram(Tensor<ScalarType, Properties...> Y,
		  const std::size_t n)
{

  using tensor_type       = Tensor<ScalarType, Properties...>;
  using memory_space      = typename tensor_type::traits::memory_space;
  using onnode_layout     = typename tensor_type::traits::onnode_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;

  // constraints
  static_assert(   std::is_same_v<onnode_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point_v<tensor_value_type>,
		   "TuckerOnNode::compute_gram: supports tensors with LayoutLeft" \
		   "and floating point scalar");

  using local_gram_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space>;
  local_gram_t localGram;

  const MPI_Comm& comm = Y.getDistribution().getProcessorGrid().getColComm(n, false);
  int numProcs;
  MPI_Comm_size(comm, &numProcs);
  if(numProcs > 1)
  {
    impl::local_gram_after_data_redistribution(Y, n, localGram);
  }
  else{
    impl::local_gram_without_data_redistribution(Y, n, localGram);
  }

  return impl::reduce_for_gram(localGram);
}

}
#endif
