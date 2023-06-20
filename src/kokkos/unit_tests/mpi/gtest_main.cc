#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc,argv);
  int err = 0;
  {
    int rank = {};
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Kokkos::initialize (argc, argv);
    {

      /*
	If MPI is on, we do not want gtest-specific prints
	(RUN, etc) see below, to show up on every rank polluting the terminal.
	We only want info about tests running and passing from a single rank like below.
	Regardless of many ranks the test uses.

	1: [==========] Running 4 tests from 1 test suite.
	1: [----------] Global test environment set-up.
	1: [----------] 4 tests from tuckermpi_map
	1: [ RUN      ] tuckermpi_map.default_constructor
	1: [       OK ] tuckermpi_map.default_constructor (0 ms)
	1: [ RUN      ] tuckermpi_map.getLocalIndex
	1: [       OK ] tuckermpi_map.getLocalIndex (0 ms)
	1: [----------] 4 tests from tuckermpi_map (0 ms total)
	1: [----------] Global test environment tear-down
	1: [==========] 4 tests from 1 test suite ran. (0 ms total)
	1: [  PASSED  ] 4 tests.

	The listener code below allows that.
	see https://github.com/google/googletest/issues/822
      */
      ::testing::TestEventListeners& listeners =
	::testing::UnitTest::GetInstance()->listeners();
      if (rank != 0) {
	delete listeners.Release(listeners.default_result_printer());
      }

      err = RUN_ALL_TESTS();
    }
    Kokkos::finalize();
    MPI_Finalize();
  }
  return err;
}
