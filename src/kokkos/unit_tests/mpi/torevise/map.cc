#include <gtest/gtest.h>
#include "mpi.h"
#include "TuckerMpi_Map.hpp"

using namespace TuckerMpi;

TEST(tuckermpidistributed, map_getLocalIndex)
{
  int globalNumEntries = 72;
  int local_index = -12;
  const MPI_Comm comm = MPI_COMM_WORLD;
  int myRank;
  MPI_Comm_rank(comm,&myRank);
  Map m(globalNumEntries, comm);
  int local_indexBegin_ = m.getOffset(myRank);

  ASSERT_EQ(m.getLocalIndex(local_index), -1);
  local_index = 231;
  ASSERT_EQ(m.getLocalIndex(local_index), -1);
  local_index = 35;
  ASSERT_EQ(m.getLocalIndex(local_index), local_index - local_indexBegin_);
}

TEST(tuckermpidistributed, map_getGlobalIndex)
{

  int globalNumEntries = 3;
  int global_index = 7;
  const MPI_Comm comm = MPI_COMM_WORLD;
  int myRank;
  MPI_Comm_rank(comm,&myRank);

  Map m(globalNumEntries, comm);

  int local_indexBegin_ = m.getOffset(myRank);

  ASSERT_EQ(m.getGlobalIndex(global_index), -1);

  global_index = 1;
  ASSERT_EQ(m.getGlobalIndex(global_index), local_indexBegin_ + global_index);
}

TEST(tuckermpidistributed, map_getLocalNumEntries)
{
  int globalNumEntries = 13;
  const MPI_Comm comm = MPI_COMM_WORLD;
  int myRank;
  MPI_Comm_rank(comm,&myRank);

  Map m(globalNumEntries, comm);

  int local_indexBegin_ = m.getOffset(myRank);
  int local_indexEnd_ = m.getOffset(myRank+1)-1;
  int local_localnumentries = 1 + local_indexEnd_ - local_indexBegin_;

  ASSERT_EQ(m.getLocalNumEntries(), local_localnumentries);
}

TEST(tuckermpidistributed, map_getGlobalNumEntries)
{
  int globalNumEntries = 10;
  const MPI_Comm comm = MPI_COMM_WORLD;

  Map m(globalNumEntries, comm);
  ASSERT_EQ(m.getGlobalNumEntries(), globalNumEntries);
}

// TOTEST: int getMaxNumEntries() const;

TEST(tuckermpidistributed, map_getNumEntries)
{
  int globalNumEntries = 62;
  const MPI_Comm comm = MPI_COMM_WORLD;
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  Map m(globalNumEntries, comm);
  ASSERT_EQ(m.getNumEntries(0), globalNumEntries/nprocs);
}

TEST(tuckermpidistributed, map_getOffset)
{
  int globalNumEntries = 300;
  const MPI_Comm comm = MPI_COMM_WORLD;
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  Map m(globalNumEntries, comm);
  ASSERT_EQ(m.getOffset(0), 0);
  ASSERT_EQ(m.getOffset(1), globalNumEntries/nprocs);
}

TEST(tuckermpidistributed, map_getComm)
{
  int globalNumEntries = 423;
  const MPI_Comm comm = MPI_COMM_WORLD;
  Map m(globalNumEntries, comm);
  ASSERT_EQ(m.getComm(), comm);
}

// TOTEST: void removeEmptyProcs();
