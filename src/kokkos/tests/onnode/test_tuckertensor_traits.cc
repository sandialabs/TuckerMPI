#include <gtest/gtest.h>
#include "TuckerOnNode_TuckerTensor.hpp"
#include <Kokkos_Core.hpp>

TEST(tuckerkokkos_tuckertensor, traits)
{
  using namespace TuckerOnNode;

  {
    using scalar_t = double;
    using ttensor_t = TuckerTensor<scalar_t>;
    using traits   = typename ttensor_t::traits;
    using expected_mem_space = Kokkos::DefaultExecutionSpace::memory_space;
    static_assert(std::is_same_v<typename traits::memory_space, expected_mem_space>, "");
    // using view_t = typename traits::data_view_type;
    // static_assert(std::is_same_v<typename view_t::traits::value_type, double>, "");
  }

  // TODO: finish
}
