#include "Tucker_CmdLineParse.hpp"
#include "MpiKokkos_Tucker_ParameterFileParser.hpp"

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
    // Get the rank of this MPI process
    // Only rank 0 will print to stdout
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // parse cmd line and param file
    const auto paramfn = parse_cmdline_or(argc, (const char**)argv,
					  "--parameter-file", "paramfile.txt");
    const InputParameters<scalar_t> inputs(paramfn);
    //int result_inputs_check_args = inputs.check_args();
    if(rank == 0) {
      //std::cout << "Argument checking: " << result_inputs_check_args << std::endl;
      inputs.describe(); 
    }
    
    // TO CONTINUE
    // SEE mpi AND kokkosonly CODES


  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}