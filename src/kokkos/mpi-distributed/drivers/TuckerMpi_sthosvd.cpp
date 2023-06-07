#include "Tucker_CmdLineParse.hpp"
#include "TuckerMpi_ParameterFileParser.hpp"
#include "TuckerMpi_Distribution.hpp"

#include <mpi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // parse cmd line and param file
    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
					  "--parameter-file", "paramfile.txt");
    const TuckerMpiDistributed::InputParameters<scalar_t> inputs(paramfn);
    if(rank == 0) {
      inputs.describe();
    }

    // set up distribution object
    TuckerMpiDistributed::Distribution dist(inputs.sizeArrayOfDataTensor(), inputs.proc_grid_dims);

    // TO CONTINUE
    // SEE mpi AND kokkosonly CODES
  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
