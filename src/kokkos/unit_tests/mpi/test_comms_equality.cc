#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;

TEST(tuckermpi_mpihelpers, comms_equal){
  MPI_Comm comm1 = MPI_COMM_WORLD;
  MPI_Comm comm2 = MPI_COMM_WORLD;
  ASSERT_TRUE( impl::comms_equal(comm1, comm2) );
}

TEST(tuckermpi_mpihelpers, comms_equal2){
  MPI_Comm comm1 = MPI_COMM_NULL;
  MPI_Comm comm2 = MPI_COMM_WORLD;
  ASSERT_FALSE( impl::comms_equal(comm1, comm2) );
}

TEST(tuckermpi_mpihelpers, comms_equal3){
  MPI_Comm comm1 = MPI_COMM_WORLD;
  MPI_Comm comm2 = MPI_COMM_NULL;
  ASSERT_FALSE( impl::comms_equal(comm1, comm2) );
}

TEST(tuckermpi_mpihelpers, comms_equal4){
  MPI_Comm comm1 = MPI_COMM_NULL;
  MPI_Comm comm2 = MPI_COMM_NULL;
  ASSERT_TRUE( impl::comms_equal(comm1, comm2) );
}

TEST(tuckermpi_mpihelpers, comms_equal5){
  MPI_Comm comm1 = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm1, &rank);

  int color = 0;
  if (rank % 2 == 0){ color = 1; };

  MPI_Comm comm2;
  MPI_Comm_split(comm1, color, 0, &comm2);

  ASSERT_FALSE( impl::comms_equal(comm1, comm2) );
}


TEST(tuckermpi_mpihelpers, sharedptr_comms_equal){
  std::shared_ptr<MPI_Comm> comm1(new MPI_Comm(MPI_COMM_WORLD));
  std::shared_ptr<MPI_Comm> comm2(new MPI_Comm(MPI_COMM_WORLD));
  ASSERT_TRUE( impl::comms_equal(comm1, comm2) );
}

TEST(tuckermpi_mpihelpers, sharedptr_comms_equal2){
  std::shared_ptr<MPI_Comm> comm1(new MPI_Comm(MPI_COMM_NULL));
  std::shared_ptr<MPI_Comm> comm2(new MPI_Comm(MPI_COMM_WORLD));
  ASSERT_FALSE( impl::comms_equal(comm1, comm2) );
}

TEST(tuckermpi_mpihelpers, sharedptr_comms_equal3){
  std::shared_ptr<MPI_Comm> comm1(new MPI_Comm(MPI_COMM_WORLD));
  std::shared_ptr<MPI_Comm> comm2(new MPI_Comm(MPI_COMM_NULL));
  ASSERT_FALSE( impl::comms_equal(comm1, comm2) );
}

TEST(tuckermpi_mpihelpers, sharedptr_comms_equal4){
  std::shared_ptr<MPI_Comm> comm1(new MPI_Comm(MPI_COMM_NULL));
  std::shared_ptr<MPI_Comm> comm2(new MPI_Comm(MPI_COMM_NULL));
  ASSERT_TRUE( impl::comms_equal(comm1, comm2) );
}
