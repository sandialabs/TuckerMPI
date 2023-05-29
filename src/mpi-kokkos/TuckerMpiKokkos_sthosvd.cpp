#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif

  // Initialize MPI
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {

    // Get the rank of this MPI process
    // Only rank 0 will print to stdout
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    //
    if(rank == 0) { std::cout << "rank:" << rank << "; nprocs: " << nprocs << std::endl; }

    std::cout << "Inside Kokkos" << std::endl;

  }
  Kokkos::finalize();
  std::cout << "After Kokkos" << std::endl;
  // Finalize MPI
  MPI_Finalize();
  std::cout << "After MPI" << std::endl;
  return 0;
}