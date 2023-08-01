#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

template<class ...Args>
using ttensor_t = Tucker::impl::TuckerTensor<true, Args...>;

TEST(tuckerkokkos_tuckertensor, traits)
{
  {
    using scalar_t = double;
    using traits   = typename ttensor_t<scalar_t>::traits;
    using expected_mem_space = Kokkos::DefaultExecutionSpace::memory_space;
    static_assert(std::is_same_v<typename traits::memory_space, expected_mem_space>, "");
  }

  // TODO: finish
}
