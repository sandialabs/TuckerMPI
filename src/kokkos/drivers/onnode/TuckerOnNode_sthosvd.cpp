
#include "CmdLineParse.hpp"
#include "ParameterFileParser.hpp"
#include "TuckerOnNode.hpp"

int main(int argc, char* argv[])
{
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif

  Kokkos::initialize(argc, argv);
  {
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

    /*
     * read data and create tensor
     */
    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
						  "--parameter-file", "paramfile.txt");
    const TuckerOnNode::InputParameters<scalar_t> inputs(paramfn);
    inputs.describe();
    TuckerOnNode::Tensor<scalar_t, memory_space> X(inputs.dimensionsOfDataTensor());
    TuckerOnNode::read_tensor_binary(X, inputs.in_fns_file.c_str());

    /*
     * preprocessing
     */
    std::cout << "Compute statistics" << std::endl;
    Tucker::compute_statistics(X, inputs.scale_mode, inputs.stats_file, inputs.stdThresh);
    // FIXME: Perform preprocessing is missingx

    /*
     * prepare lambdas "expressing" the computation to do
     */
    auto writeCoreTensorToFile = [=](auto factorization)
    {
      const std::string coreFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_core.mpi";
      std::cout << "Writing core tensor to " << coreFilename << std::endl;
      TuckerOnNode::export_tensor_binary(factorization.coreTensor(), coreFilename.c_str());
    };

    auto printNorms = [=](auto factorization){
      const auto xnorm = std::sqrt(X.frobeniusNormSquared());
      const auto gnorm = std::sqrt(factorization.coreTensor().frobeniusNormSquared());
      std::cout << "Norm of input tensor: " << std::setprecision(7) << xnorm << std::endl;
      std::cout << "Norm of core tensor: " << std::setprecision(7) << gnorm << std::endl;
    };

    auto truncator =
      Tucker::create_core_tensor_truncator(X, inputs.dimensionsOfCoreTensor(), inputs.tol);

    auto sthosvdGram = [=](auto truncator){
      const auto method = TuckerOnNode::Method::Gram;
      auto [tt, eigvals] = TuckerOnNode::sthosvd(method, X, truncator, false /*flipSign*/);

      writeCoreTensorToFile(tt);

      const std::string filePrefix = inputs.sv_dir + "/" + inputs.sv_fn + "_mode_";
      Tucker::print_eigenvalues(eigvals, filePrefix, false /*for gram we write raw eigenvalues*/);

      printNorms(tt);
    };

    /*
     * run for real
     */
    if(inputs.boolSTHOSVD){
      if (!inputs.boolUseLQ){
	sthosvdGram(truncator);
      }
    }

  } // local scope to ensure all Kokkos views are destructed appropriately
  Kokkos::finalize();

  return 0;
}
