#include "Tucker_CmdLineParse.hpp"
#include "TuckerMpi_ParameterFileParser.hpp"
#include "TuckerMpi_Distribution.hpp"
#include "TuckerMpi_Tensor.hpp"
#include "TuckerMpi_Tensor_IO.hpp"
#include "TuckerMpi_CoreTensorTruncator.hpp"
#include "TuckerMpi_sthosvd.hpp"
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <unistd.h>

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
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
					  "--parameter-file", "paramfile.txt");
    const TuckerMpiDistributed::InputParameters<scalar_t> inputs(paramfn);
    if(rank == 0) { inputs.describe(); }

    const auto dataTensorDim = inputs.dimensionsOfDataTensor();
    TuckerMpi::Distribution dist(dataTensorDim, inputs.proc_grid_dims);
    TuckerMpi::Tensor<scalar_t, memory_space> X(std::move(dist));
    TuckerMpi::read_tensor_binary(X, inputs.in_fns_file.c_str());
#if 0
    /*FRIZZI: tmp debug print*/
    auto v = X.getLocalTensor().data();
    sleep(rank*1);
    std::cout << "Rank = " << rank << " ";
    for (int i=0; i<v.extent(0); ++i){ std::cout << v(i) << " "; }
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
    /**/
#endif

    if(rank == 0) {
      const size_t local_nnz = X.getLocalNumEntries();
      const size_t global_nnz = X.getGlobalNumEntries();
      std::cout << "Local input tensor size: ";
      Tucker::write_view_to_stream(std::cout, X.getLocalSize());
      Tucker::printBytes(local_nnz*sizeof(scalar_t));

      std::cout << "Global input tensor size: ";
      Tucker::write_view_to_stream(std::cout, X.getGlobalSize());
      Tucker::printBytes(global_nnz*sizeof(scalar_t));
    }

    // FIXME: Compute statistics is missing
    // FIXME: Perform preprocessing is missing

    auto coreTensorTruncator =
      TuckerMpi::create_core_tensor_truncator(X, inputs.dimensionsOfCoreTensor(), inputs.tol);

    bool flipSign = false;
    if(inputs.boolSTHOSVD){
      auto f = TuckerMpi::STHOSVD(X, coreTensorTruncator, inputs.modeOrder,
				  inputs.boolUseOldGram, flipSign,
				  inputs.boolUseLQ, inputs.useButterflyTSQR);
    }

  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
