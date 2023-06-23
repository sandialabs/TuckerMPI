#include <gtest/gtest.h>
#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>

namespace {

template<class T, class ... Ps>
void assert_rank1_view_has_all_zeros(Kokkos::View<T, Ps...> v)
{
  static_assert(v.rank == 1);
  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  for (std::size_t i=0; i<v_h.extent(0); ++i){
    ASSERT_TRUE(v_h(i) == 0);
  }
}

} // anonyn namespace

using namespace TuckerOnNode;
using scalar_t = double;

TEST(tuckerkokkos_tensor, constructors){
  Tensor<scalar_t> x;
  Tensor<scalar_t> y{2,3};
  Tensor<scalar_t> z(std::vector<int>{4,5});
}

TEST(tuckerkokkos_tensor, rank){
  Tensor<scalar_t> a;
  Tensor<scalar_t> b{1};
  Tensor<scalar_t> c{1,2};
  Tensor<scalar_t> d{3,4,5};
  ASSERT_TRUE(a.rank() == -1);
  ASSERT_TRUE(b.rank() == 1);
  ASSERT_TRUE(c.rank() == 2);
  ASSERT_TRUE(d.rank() == 3);
}

TEST(tuckerkokkos_tensor, size){
  Tensor<scalar_t> a;
  Tensor<scalar_t> b{5};
  Tensor<scalar_t> c{1,2};
  Tensor<scalar_t> d{3,4,5};
  ASSERT_TRUE(a.size() == 0);
  ASSERT_TRUE(b.size() == 5);
  ASSERT_TRUE(c.size() == 2);
  ASSERT_TRUE(d.size() == 60);
}

TEST(tuckerkokkos_tensor, extent){
  Tensor<scalar_t> b{5};
  Tensor<scalar_t> c{1,2};
  Tensor<scalar_t> d{3,4,5};
  ASSERT_TRUE(b.extent(0) == 5);
  ASSERT_TRUE(c.extent(0) == 1);
  ASSERT_TRUE(c.extent(1) == 2);
  ASSERT_TRUE(d.extent(0) == 3);
  ASSERT_TRUE(d.extent(1) == 4);
  ASSERT_TRUE(d.extent(2) == 5);
}

TEST(tuckerkokkos_tensor, postconditions_default_constructor){
  Tensor<scalar_t> x;
  ASSERT_TRUE(x.rank() == -1);
  ASSERT_TRUE(x.size() == 0);
  auto view = x.data();
  ASSERT_TRUE(view.extent(0) == 0);
}

TEST(tuckerkokkos_tensor, postcondition_constructor_emptylist){
  Tensor<scalar_t> x{};
  ASSERT_TRUE(x.rank() == -1);
  ASSERT_TRUE(x.size() == 0);
  auto view = x.data();
  ASSERT_TRUE(view.extent(0) == 0);
}

TEST(tuckerkokkos_tensor, postcondition_constructor_rank1){
  Tensor<scalar_t> x{2};
  ASSERT_TRUE(x.rank() == 1);
  ASSERT_TRUE(x.size() == 2);
  ASSERT_TRUE(x.extent(0) == 2);
  auto view = x.data();
  ASSERT_TRUE(view.extent(0) == 2);
  assert_rank1_view_has_all_zeros(view);
}

TEST(tuckerkokkos_tensor, postcondition_constructor_rank2){
  Tensor<scalar_t> x{2,3};
  ASSERT_TRUE(x.rank() == 2);
  ASSERT_TRUE(x.size() == 6);
  ASSERT_TRUE(x.extent(0) == 2);
  ASSERT_TRUE(x.extent(1) == 3);
  auto view = x.data();
  ASSERT_TRUE(view.extent(0) == 6);
  assert_rank1_view_has_all_zeros(view);
}

TEST(tuckerkokkos_tensor, postcondition_constructor_rank3){
  Tensor<scalar_t> x{2,3,5};
  ASSERT_TRUE(x.rank() == 3);
  ASSERT_TRUE(x.size() == 30);
  ASSERT_TRUE(x.extent(0) == 2);
  ASSERT_TRUE(x.extent(1) == 3);
  ASSERT_TRUE(x.extent(2) == 5);
  auto view = x.data();
  ASSERT_TRUE(view.extent(0) == 30);
  assert_rank1_view_has_all_zeros(view);
}

TEST(tuckerkokkos_tensor, copy_constructor_shallow_copy)
{
  Tensor<scalar_t> x({2,1,5});
  auto x_view = x.data();
  Kokkos::Random_XorShift64_Pool<> pool(4543423);
  Kokkos::fill_random(x_view, pool, -1., 1.);
  auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_view);

  Tensor<scalar_t> y = x;
  auto y_view = y.data();
  ASSERT_TRUE(x.dimensions() == y.dimensions());
  auto y_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_view);
  for (std::size_t i=0; i<y_h.extent(0); ++i){
    ASSERT_TRUE(x_h(i) == y_h(i));
  }
}

TEST(tuckerkokkos_tensor, copy_assign_mismatching_ranks)
{
  Tensor<scalar_t> x({2,5});
  Tensor<scalar_t> y({2,5,3});
  EXPECT_THROW(
	       try{
		 y = x;
	       }
	       catch(const std::runtime_error & ex){
		 EXPECT_STREQ("Tensor: mismatching ranks for copy assignemnt", ex.what());
		 throw;
	       },
	       std::runtime_error);
}

TEST(tuckerkokkos_tensor, copy_assign_different_dims)
{
  Tensor<scalar_t> x({2,5,1});
  Tensor<scalar_t> y({2,5,3});
  y = x;
  ASSERT_TRUE(y.rank() == 3);
  ASSERT_TRUE(y.size() == 10);
  ASSERT_TRUE(y.extent(0) == 2);
  ASSERT_TRUE(y.extent(1) == 5);
  ASSERT_TRUE(y.extent(2) == 1);
}

TEST(tuckerkokkos_tensor, copy_assign_shallow_copy){
  Tensor<scalar_t> x({2,1,5});
  auto x_view = x.data();
  Kokkos::Random_XorShift64_Pool<> pool(4543423);
  Kokkos::fill_random(x_view, pool, -1., 1.);
  auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_view);

  Tensor<scalar_t> y;
  y = x;
  auto y_view = y.data();
  ASSERT_TRUE(x.dimensions() == y.dimensions());
  auto y_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_view);
  for (std::size_t i=0; i<y_h.extent(0); ++i){
    ASSERT_TRUE(x_h(i) == y_h(i));
  }
}

TEST(tuckerkokkos_tensor, copy_constr_const_semantics){
  Tensor<scalar_t, Kokkos::HostSpace> x({2,1,5});
  auto x_view = x.data();
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(4543423);
  Kokkos::fill_random(x_view, pool, -1., 1.);

  Tensor<const scalar_t, Kokkos::HostSpace> y(x);
  auto y_view = y.data();
  //y_view(0) = 4.; // this SHOULD NOT compile

  // FIXME: add test to ensure this line gives compile error similar to:
  // error: assignment of read-only location 'y_view.Kokkos::View<const double*, Kokkos::LayoutLeft, Kokkos::HostSpace>::operator()<int>(0)'
}

TEST(tuckerkokkos_tensor, copy_assign_const_semantics){
  Tensor<scalar_t, Kokkos::HostSpace> x({2,1,5});
  auto x_view = x.data();
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(4543423);
  Kokkos::fill_random(x_view, pool, -1., 1.);

  Tensor<const scalar_t, Kokkos::HostSpace> y;
  y = x;
  auto y_view = y.data();
  //y_view(0) = 4.; // this SHOULD NOT compile

  // FIXME: add test to ensure this line gives compile error similar to:
  // error: assignment of read-only location 'y_view.Kokkos::View<const double*, Kokkos::LayoutLeft, Kokkos::HostSpace>::operator()<int>(0)'
}
