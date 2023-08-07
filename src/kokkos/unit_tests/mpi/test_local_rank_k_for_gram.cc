#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;

const MPI_Comm comm = MPI_COMM_WORLD;

int mpi_rank(){
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

void setup_and_run(int mode){
  using scalar_t = double;

  std::vector<int> rows = {5, 6, 3, 4};
  std::vector<int> cols = {25, 26, 23, 24};
  
  impl::Matrix<scalar_t> M(rows[mpi_rank()], cols[mpi_rank()], comm, false);
  
  Kokkos::Random_XorShift64_Pool<> pool(1234567);
  Kokkos::fill_random(M.getLocalMatrix(), pool, 10);
  
  Kokkos::View<scalar_t**, Kokkos::LayoutLeft> gold =
    impl::local_rank_k_for_gram_host(M, mode, 4);
  
  Kokkos::View<scalar_t**, Kokkos::LayoutLeft> computed =
    impl::local_rank_k_for_gram(M, mode, 4);
  
  auto gold_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gold);
  auto computed_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), computed);
  for (std::size_t j=0; j<gold_h.extent(1); ++j){
    for (std::size_t i=0; i<gold_h.extent(0); ++i){
      EXPECT_NEAR(gold_h(i,j), computed_h(i,j), 0.001);
    }
  }
}

TEST(tuckermpi_local_rank_k_for_gram, test_along_mode_0){
  setup_and_run(0);
}

TEST(tuckermpi_local_rank_k_for_gram, test_along_mode_1){
  setup_and_run(1);
}

TEST(tuckermpi_local_rank_k_for_gram, test_along_mode_2){
  setup_and_run(2);
}

TEST(tuckermpi_local_rank_k_for_gram, test_along_mode_3){
  setup_and_run(3);
}