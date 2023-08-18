#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;

const MPI_Comm comm = MPI_COMM_WORLD;

int mpi_rank(){
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

TEST(tuckermpi_map, default_constructor){
  Map m;
}

TEST(tuckermpi_map, default_constructor_postconditions){
  Map m;

  ASSERT_EQ(m.getLocalNumEntries(),  0);
  ASSERT_EQ(m.getGlobalNumEntries(), 0);
  ASSERT_EQ(m.getMaxNumEntries(), 0);

  for (int r=0; r<3; ++r){
    ASSERT_EQ(m.getNumEntries(r), 0);
    ASSERT_EQ(m.getOffset(r), 0);
  }
  ASSERT_TRUE(m.getComm() == MPI_COMM_NULL);
}

TEST(tuckermpi_map, constructor){
  Map m(9, comm);
}

TEST(tuckermpi_map, getLocalIndex){
  Map m(9, comm);

  ASSERT_EQ(m.getLocalNumEntries(),  3);
  ASSERT_EQ(m.getGlobalNumEntries(), 9);

  if (mpi_rank() == 0){
    ASSERT_EQ(m.getLocalIndex(0), 0);
    ASSERT_EQ(m.getLocalIndex(1), 1);
    ASSERT_EQ(m.getLocalIndex(2), 2);
  }

  if (mpi_rank() == 1){
    ASSERT_EQ(m.getLocalIndex(3), 0);
    ASSERT_EQ(m.getLocalIndex(4), 1);
    ASSERT_EQ(m.getLocalIndex(5), 2);
  }

  if (mpi_rank() == 2){
    ASSERT_EQ(m.getLocalIndex(6), 0);
    ASSERT_EQ(m.getLocalIndex(7), 1);
    ASSERT_EQ(m.getLocalIndex(8), 2);
  }
}


TEST(tuckermpi_map, hasGlobalIndex){
  Map m(10, comm);
  using v_t = std::vector<int>;

  const auto lambda = [&](int i) -> bool {
    return m.hasGlobalIndex(i); };

  if (mpi_rank() == 0){
    v_t has_v = {0,1,2,3};
    v_t not_v = {4,5,6,7,11,20};
    ASSERT_TRUE(std::all_of(has_v.begin(), has_v.end(), lambda));
    ASSERT_TRUE(std::none_of(not_v.begin(), not_v.end(), lambda));
  }
  else if (mpi_rank() == 1){
    v_t has_v = {4,5,6};
    v_t not_v = {0,1,2,3,7,11,20};
    ASSERT_TRUE(std::all_of(has_v.begin(), has_v.end(), lambda));
    ASSERT_TRUE(std::none_of(not_v.begin(), not_v.end(), lambda));
  }
  else if (mpi_rank() == 2){
    v_t has_v = {7,8,9};
    v_t not_v = {0,1,2,3,4,5,6,10,11,20,120};
    ASSERT_TRUE(std::all_of(has_v.begin(), has_v.end(), lambda));
    ASSERT_TRUE(std::none_of(not_v.begin(), not_v.end(), lambda));
  }
}
