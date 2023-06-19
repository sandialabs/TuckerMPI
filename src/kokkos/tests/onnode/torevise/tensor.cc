#include <gtest/gtest.h>
#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_SizeArray.hpp"
#include <Kokkos_Core.hpp>

TEST(tuckerkokkos, tensor_traits)
{
  using namespace TuckerOnNode;

  {
    using scalar_t = double;
    using tensor_t = Tensor<scalar_t>;
    using traits = typename tensor_t::traits;
    using expected_mem_space =Kokkos::DefaultExecutionSpace::memory_space;
    static_assert(std::is_same_v<typename traits::memory_space, expected_mem_space>, "");
    using view_t = typename traits::view_type;
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

    using view_t = typename traits::view_type;
    static_assert(std::is_same_v<typename view_t::traits::value_type, float>, "");
  }
}

TEST(tuckerkokkos, tensor_constructor)
{
  using namespace TuckerOnNode;
  using scalar_t = double;
  Tucker::SizeArray sa(3);
  sa[0] = 2;
  sa[1] = 1;
  sa[2] = 5;
  Tensor<scalar_t> X(sa);
  // we should do something better here
}

TEST(tuckerkokkos, tensor_constructor_view_zeros)
{
  using namespace TuckerOnNode;
  using scalar_t = double;
  Tucker::SizeArray sa(3);
  sa[0] = 2;
  sa[1] = 1;
  sa[2] = 5;
  Tensor<scalar_t> x(sa);
  auto d_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x.data());
  for (std::size_t i=0; i<d_h.extent(0); ++i){
    ASSERT_DOUBLE_EQ(d_h(i), 0.);
  }
}

TEST(tuckerkokkos, tensor_copy_constructor_shallow_copy)
{
  using namespace TuckerOnNode;
  using scalar_t = double;
  Tucker::SizeArray sa(3);
  sa[0] = 2;
  sa[1] = 1;
  sa[2] = 5;
  Tensor<scalar_t> x(sa);
  x.fillRandom(1., 5.);
  auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x.data());

  // shallow copy constructor, y should have same data as x
  Tensor<scalar_t> y = x;
  auto y_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y.data());
  ASSERT_TRUE(x_h.extent(0) == y_h.extent(0));
  for (std::size_t i=0; i<y_h.extent(0); ++i){
    ASSERT_TRUE(x_h(i) == y_h(i));
  }
}

TEST(tuckerkokkos, tensor_copy_assign_shallow_copy)
{
  using namespace TuckerOnNode;
  using scalar_t = double;
  Tucker::SizeArray sa(3);
  sa[0] = 2; sa[1] = 1; sa[2] = 5;
  Tensor<scalar_t> x(sa);
  x.fillRandom(1., 5.);
  auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x.data());

  // shallow copy constructor, y should have same data as x
  Tensor<scalar_t> y;
  y = x;
  auto y_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y.data());
  ASSERT_TRUE(x_h.extent(0) == y_h.extent(0));
  for (std::size_t i=0; i<y_h.extent(0); ++i){
    ASSERT_TRUE(x_h(i) == y_h(i));
  }
}

TEST(tuckerkokkos, tensor_N)
{
  using namespace TuckerOnNode;
  using scalar_t = double;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  Tucker::SizeArray sa(3);
  Tensor<scalar_t, memory_space> x(sa);
  ASSERT_EQ(x.rank(), 3);
}

TEST(tuckerkokkos, tensor_size)
{
  using namespace TuckerOnNode;
  using scalar_t = double;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  Tucker::SizeArray sa(3);
  sa[0] = 5; sa[1] = 7; sa[2] = 9;
  Tensor<scalar_t, memory_space> x(sa);
  ASSERT_EQ(x.extent(0), 5);
  ASSERT_EQ(x.extent(1), 7);
  ASSERT_EQ(x.extent(2), 9);
}

TEST(tuckerkokkos, tensor_frobeniusNormSquared)
{
  using namespace TuckerOnNode;
  using scalar_t = double;
  Tucker::SizeArray sa(3);
  sa[0] = 2; sa[1] = 3; sa[2] = 4;
  // tensor
  Tensor<scalar_t> x(sa);
  x.fillRandom(1., 5.);
  auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x.data());
  // vector with same data
  scalar_t x_h_extent = x_h.extent(0);
  std::vector<scalar_t> v(x_h_extent);
  for (std::size_t i=0; i<x_h_extent; ++i){ v[i] = x_h[i]; }
  // do 2 norm of this vector
  scalar_t norm = 0.;
  for(std::size_t i=0; i<x_h_extent; ++i){ norm += v[i] * v[i]; }
  EXPECT_NEAR(x.frobeniusNormSquared(), norm, 0.001);
}