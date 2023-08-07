#include "CmdLineParse.hpp"
#include "ParameterFileParser.hpp"
#include "TuckerMpi.hpp"

template<class ScalarType>
void run(const TuckerMpiDistributed::InputParameters<ScalarType> & inputs)
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
  if(mpiRank == 0) { inputs.describe(); }

  const auto dataTensorDim = inputs.dimensionsOfDataTensor();
  TuckerMpi::Tensor<ScalarType, memory_space> X(dataTensorDim, inputs.proc_grid_dims);
  TuckerMpi::read_tensor_binary(X, inputs.rawDataFilenames);

  if(mpiRank == 0) {
    const size_t local_nnz = X.localSize();
    const size_t global_nnz = X.globalSize();
    std::cout << "Local input tensor size  : ";
    Tucker::write_view_to_stream_singleline(std::cout, X.localDimensionsOnHost());
    std::cout << ", or "; Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(ScalarType));
    std::cout << "Global input tensor size : ";
    Tucker::write_view_to_stream_singleline(std::cout, X.globalDimensionsOnHost());
    std::cout << ", or "; Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(ScalarType));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /*
   * preprocessing
   */
  // FIXME: Compute statistics is missing
  // FIXME: Perform preprocessing is missing

  /*
   * prepare lambdas "expressing" the computation to do
   */
  auto writeEigenvaluesToFile = [=](auto container){
    if(mpiRank == 0) {
      const std::string filePrefix = inputs.sv_dir + "/" + inputs.sv_fn + "_mode_";

      const int nmodes = container.rank();
      for(int mode=0; mode<nmodes; mode++){
	std::ostringstream ss;
	ss << filePrefix << mode << ".txt";
	std::ofstream ofs(ss.str());
	std::cout << "Writing singular values to " << ss.str() << std::endl;

	auto eigvals = container.eigenvalues(mode);
	auto eigvals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigvals);
	for(int i=0; i<eigvals.extent(0); i++) {
	  ofs << std::setprecision(16) << eigvals_h(i) << std::endl;
	}
	ofs.close();
      }
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
    auto [tt, eigvals] = TuckerMpi::sthosvd(method, X, truncator,
					    inputs.modeOrder, false /*flipSign*/);
    writeEigenvaluesToFile(eigvals);
    printNorms(tt);
  };

  /*
   * run for real
   */
  if(inputs.boolSTHOSVD){
    sthosvdNewGram(truncator);
  }
}

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
    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
						  "--parameter-file", "paramfile.txt");
    const TuckerMpiDistributed::InputParameters<scalar_t> inputs(paramfn);
    run(inputs);
  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
