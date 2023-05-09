#include <gtest/gtest.h>
#include "Tucker_Tensor.hpp"
#include "Tucker_SizeArray.hpp"
#include <Kokkos_Core.hpp>

TEST(tuckerkokkos, tensor_N)
{
  using namespace TuckerKokkos;

  using scalar_t = double;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  SizeArray sa(3);
  Tensor<scalar_t, memory_space> X(sa);
  ASSERT_EQ(X.N(), 3);
}

TEST(tuckerkokkos, tensor_size)
{
  using namespace TuckerKokkos;
  using scalar_t = double;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  SizeArray sa(3);
  sa[0] = 5;
  sa[1] = 7;
  sa[2] = 9;

  Tensor<scalar_t, memory_space> X(sa);
  ASSERT_EQ(X.size(0), 5);
  ASSERT_EQ(X.size(1), 7);
  ASSERT_EQ(X.size(2), 9);
}
