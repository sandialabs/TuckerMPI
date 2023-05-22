
#include <Kokkos_Core.hpp>
#include "Tucker_IO_Util.hpp"
#include "init_args.hpp"
#include "Tucker.hpp"
#include <variant>

int main(int argc, char* argv[])
{
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif

  Kokkos::initialize();
  {
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

    //
    // parsing
    //
    const std::string paramfn =
      TuckerKokkos::parseString(argc, (const char**)argv,
			  "--parameter-file", "paramfile.txt");
    const std::vector<std::string> fileAsString = TuckerKokkos::getFileAsStrings(paramfn);
    InputArgs args = parse_input_file<scalar_t>(fileAsString);
    int checkArgs = check_args(args);
    std::cout << "Argument checking: passed" << std::endl;
    print_args(args);

    chech_array_sizes(args);
    std::cout << "Array sizes checking: passed" << std::endl;

    const auto I_dims = TuckerKokkos::stringParseSizeArray(fileAsString, "Global dims");
    args.nd = I_dims.size();
    std::cout << "The global dimensions of the tensor to be scaled or compressed\n";
    std::cout << "- Global dims = " << I_dims << std::endl << std::endl;

    using core_tensor_rank_finding_strategies = std::variant<
      TuckerKokkos::CoreRankUserDefined,
      TuckerKokkos::CoreRankViaThreshold<scalar_t>>;
    core_tensor_rank_finding_strategies coreTensorRankStrategy;
    if (!args.boolAuto) {
      auto R_dims = TuckerKokkos::stringParseSizeArray(fileAsString, "Ranks");
      coreTensorRankStrategy = TuckerKokkos::CoreRankUserDefined{R_dims};
      std::cout << "Global dimensions of the core tensor is fixed:\n";
      std::cout << "- Ranks = " << R_dims << std::endl << std::endl;
    }
    else{
      std::cout << "Automatic rank determination of core tensor is enabled\n";
      coreTensorRankStrategy = TuckerKokkos::CoreRankViaThreshold<scalar_t>{args.tol};
    }

    //
    // reading data
    //
    TuckerKokkos::Tensor<scalar_t, memory_space> X(I_dims);
    TuckerKokkos::readTensorBinary(X, args.in_fns_file.c_str());

    //
    // compute
    //
    // FIXME: Compute statistics is missing
    // FIXME: Perform preprocessing is missing
    if(args.boolSTHOSVD)
    {
      auto f = TuckerKokkos::STHOSVD(X, coreTensorRankStrategy, args.boolUseLQ);

      // Write the eigenvalues to files
      std::string filePrefix = args.sv_dir + "/" + args.sv_fn + "_mode_";
      TuckerKokkos::printEigenvalues(f, filePrefix, false);

      printf("\n");
      const double xnorm = std::sqrt(X.norm2Squared());
      const double gnorm = std::sqrt(f.getG().norm2Squared());
      std::cout << "Norm of input tensor: " << std::setprecision(7) << xnorm << std::endl;
      std::cout << "Norm of core tensor: " << std::setprecision(7) << gnorm << std::endl;

      std::string coreFilename = args.sthosvd_dir + "/" + args.sthosvd_fn + "_core.mpi";
      std::cout << "Writing core tensor to " << coreFilename << std::endl;
      TuckerKokkos::exportTensorBinary(f.getG(), coreFilename.c_str());
    }

  } // local scope for kokkos

  Kokkos::finalize();
  return 0;
}
