#include "CmdLineParse.hpp"
#include "ParameterFileParser.hpp"
#include "TuckerMpi.hpp"

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
    /*
     * prepare
     */
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    int mpiRank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /*
     * read data and create tensor
     */
    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
						  "--parameter-file", "paramfile.txt");
    const TuckerMpiDistributed::InputParameters<scalar_t> inputs(paramfn);
    if(mpiRank == 0) { inputs.describe(); }

    const auto dataTensorDim = inputs.dimensionsOfDataTensor();
    TuckerMpi::Tensor<scalar_t, memory_space> X(dataTensorDim, inputs.proc_grid_dims);
    TuckerMpi::read_tensor_binary(X, inputs.in_fns_file.c_str());

    if(mpiRank == 0) {
      const size_t local_nnz = X.localSize();
      const size_t global_nnz = X.globalSize();
      std::cout << "Local input tensor size  : ";
      Tucker::write_view_to_stream_inline(std::cout, X.localDimensionsOnHost());
      std::cout << ", or "; Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(scalar_t));
      std::cout << "Global input tensor size : ";
      Tucker::write_view_to_stream_inline(std::cout, X.globalDimensionsOnHost());
      std::cout << ", or "; Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(scalar_t));
    }

    /*
     * preprocessing
     */
    // FIXME: Compute statistics is missing
    // FIXME: Perform preprocessing is missing
    MPI_Barrier(MPI_COMM_WORLD);

    /*
     * prepare lambdas "expressing" the computation to do
     */
    auto writeEigenvaluesToFile = [=](auto factorization, bool squareBeforeWriting){
      if(mpiRank == 0) {
	const std::string filePrefix = inputs.sv_dir + "/" + inputs.sv_fn + "_mode_";
	Tucker::print_eigenvalues(factorization, filePrefix, squareBeforeWriting);
      }
    };

    auto printNorms = [=](auto factorization){
      auto xnorm2 = X.frobeniusNormSquared();
      auto xnorm  = std::sqrt(xnorm2);
      auto gnorm  = std::sqrt(factorization.coreTensor().frobeniusNormSquared());
      if(mpiRank == 0) {
	std::cout << "Norm of input tensor: " << std::setprecision(15) << xnorm << std::endl;
	std::cout << "Norm of core tensor: "  << std::setprecision(15) << gnorm << std::endl;
      }
    };

    auto truncator =
      Tucker::create_core_tensor_truncator(X, inputs.dimensionsOfCoreTensor(), inputs.tol, mpiRank);

    auto sthosvdNewGram = [=](auto truncator){
      const auto method = TuckerMpi::Method::NewGram;
      auto f = TuckerMpi::STHOSVD(method, X, truncator,
				  inputs.modeOrder, false /*flipSign*/);
      writeEigenvaluesToFile(f, false /*for gram we write raw eigenvalues*/);
      printNorms(f);
    };

    /*
     * run for real
     */
    if(inputs.boolSTHOSVD){
      if (!inputs.boolUseOldGram && !inputs.boolUseLQ){
	sthosvdNewGram(truncator);
      }
    }

  }//end kokkos scope
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
