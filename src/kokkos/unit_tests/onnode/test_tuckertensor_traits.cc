#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

TEST(tuckerkokkos_tuckertensor, traits)
{
  {
    using scalar_t = double;
    using tensor_t = TuckerOnNode::Tensor<scalar_t>;

    using tucker_tensor_t = Tucker::TuckerTensor<tensor_t>;
    using traits   = typename tucker_tensor_t::traits;
    using expected_mem_space = Kokkos::DefaultExecutionSpace::memory_space;
    static_assert(std::is_same_v<typename traits::memory_space, expected_mem_space>, "");
  }

  // TODO: finish
}
