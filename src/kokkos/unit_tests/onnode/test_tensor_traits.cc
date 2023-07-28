#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

TEST(tuckerkokkos_tensor, traits)
{
  using namespace TuckerOnNode;

  {
    using scalar_t = double;
    using tensor_t = Tensor<scalar_t>;
    using traits   = typename tensor_t::traits;
    using expected_mem_space = Kokkos::DefaultExecutionSpace::memory_space;
    static_assert(std::is_same_v<typename traits::memory_space, expected_mem_space>, "");
    using view_t = typename traits::data_view_type;
    static_assert(std::is_same_v<typename view_t::traits::value_type, double>, "");
  }

  {
    using scalar_t = float;
    using memspace = Kokkos::HostSpace;
    using tensor_t = Tensor<scalar_t, memspace>;
    using traits = typename tensor_t::traits;
    static_assert(std::is_same_v<typename traits::memory_space, Kokkos::HostSpace>, "");
#ifdef KOKKOS_ENABLE_CUDA
    static_assert(!std::is_same_v<typename traits::memory_space, Kokkos::CudaSpace>, "");
#endif
#ifdef KOKKOS_ENABLE_HIP
    static_assert(!std::is_same_v<typename traits::memory_space, Kokkos::HIPSpace>, "");
#endif

    using view_t = typename traits::data_view_type;
    static_assert(std::is_same_v<typename view_t::traits::value_type, float>, "");
  }
}
