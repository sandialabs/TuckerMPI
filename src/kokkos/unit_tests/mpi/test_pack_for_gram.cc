#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;

const MPI_Comm comm = MPI_COMM_WORLD;

void setup_and_run(int mode){
  using scalar_t = double;
  
  // Tensor T
  // TODO: warning kokkos view here

  // Map M
  const int ndims = T.rank();
  const auto & sz = T.localDimensionsOnHost();
  const int nrows = T.globalExtent(n);
  size_t ncols = impl::prod(sz, 0, n-1, 1) * impl::prod(sz, n+1, ndims-1, 1);
  ASSERT_TRUE(ncols <= std::numeric_limits<int>::max());
  result_t recvY(nrows, (int)ncols, comm, false);
  const Map* M = recvY.getMap();
  
  std::vector<scalar_t> gold = impl::pack_for_gram_host(T, mode, M);
  std::vector<scalar_t> computed = impl::pack_for_gram(T, mode, M);

  for (int i=0; i<gold.size(); ++i){
    ASSERT_EQUAL(gold[i], computed[i]);
    //EXPECT_NEAR(gold(i), computed(j), 0.001);
  }
}

TEST(tuckermpi_pack_for_gram, test_along_mode_0){
  setup_and_run(0);
}