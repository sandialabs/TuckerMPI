
#include "Tucker_CmdLineParse.hpp"
#include "Tucker_BoilerPlate_IO.hpp"
#include "TuckerOnNode_ParameterFileParser.hpp"
#include "TuckerOnNode_CoreTensorTruncator.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include "TuckerOnNode_Tensor_IO.hpp"
#include "TuckerOnNode_sthosvd.hpp"
#include <Kokkos_Core.hpp>

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

    // parse cmd line and param file
    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
					  "--parameter-file", "paramfile.txt");
    const TuckerOnNode::InputParameters<scalar_t> inputs(paramfn);
    inputs.describe();

    // reading data
    TuckerOnNode::Tensor<scalar_t, memory_space> X(inputs.dimensionsOfDataTensor());
    TuckerOnNode::read_tensor_binary(X, inputs.in_fns_file.c_str());

    // truncator for core tensor
    auto coreTensorTruncator =
      TuckerOnNode::create_core_tensor_truncator(X, inputs.dimensionsOfCoreTensor(), inputs.tol);

    // compute
    // FIXME: Compute statistics is missing
    // FIXME: Perform preprocessing is missing
    if(inputs.boolSTHOSVD){
      auto f = TuckerOnNode::STHOSVD(X, coreTensorTruncator, inputs.boolUseLQ);

      const auto xnorm = std::sqrt(X.frobeniusNormSquared());
      const auto gnorm = std::sqrt(f.coreTensor().frobeniusNormSquared());
      std::cout << "Norm of input tensor: " << std::setprecision(7) << xnorm << std::endl;
      std::cout << "Norm of core tensor: " << std::setprecision(7) << gnorm << std::endl;

      const std::string filePrefix = inputs.sv_dir + "/" + inputs.sv_fn + "_mode_";
      TuckerOnNode::print_eigenvalues(f, filePrefix, false);
      printf("\n");
      const std::string coreFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_core.mpi";
      std::cout << "Writing core tensor to " << coreFilename << std::endl;
      TuckerOnNode::export_tensor_binary(f.coreTensor(), coreFilename.c_str());
    }

  } // local scope to ensure all Kokkos views are destructed appropriately
  Kokkos::finalize();

  return 0;
}
