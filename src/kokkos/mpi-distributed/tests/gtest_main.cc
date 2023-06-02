#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <memory>
#include "mpi.h"

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc,argv);
  int err = 0;
  {
    MPI_Init(&argc, &argv);
    Kokkos::initialize (argc, argv);
    {
      err = RUN_ALL_TESTS();
    }
    Kokkos::finalize();
    MPI_Finalize();
  }
  return err;
}