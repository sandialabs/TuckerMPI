
#include "Tucker_CmdLineParse.hpp"
#include "Tucker_ParameterFileParser.hpp"
#include "Tucker_CoreTensorTruncator.hpp"
#include "Tucker_Tensor.hpp"
#include "Tucker_BoilerPlate_IO.hpp"
#include "Tucker_sthosvd.hpp"
#include <Kokkos_Core.hpp>
#include "Tucker_IO_Util.hpp"

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
    const auto paramfn = parse_cmdline_or(argc, (const char**)argv,
					  "--parameter-file", "paramfile.txt");
    const InputParameters<scalar_t> inputs(paramfn);
    inputs.describe();

    // reading data
    TuckerKokkos::Tensor<scalar_t, memory_space> X(inputs.sizeArrayOfDataTensor());
    TuckerKokkos::readTensorBinary(X, inputs.in_fns_file.c_str());

    // truncator for core tensor
    auto coreTensorTruncator =
      TuckerKokkos::create_core_tensor_truncator(X, inputs.sizeArrayOfCoreTensor(), inputs.tol);

    // compute
    // FIXME: Compute statistics is missing
    // FIXME: Perform preprocessing is missing
    if(inputs.boolSTHOSVD){
      auto f = TuckerKokkos::STHOSVD(X, coreTensorTruncator, inputs.boolUseLQ);

      std::string filePrefix = inputs.sv_dir + "/" + inputs.sv_fn + "_mode_";
      TuckerKokkos::printEigenvalues(f, filePrefix, false);
      printf("\n");
      const auto xnorm = std::sqrt(X.frobeniusNormSquared());
      const auto gnorm = std::sqrt(f.getG().frobeniusNormSquared());
      std::cout << "Norm of input tensor: " << std::setprecision(7) << xnorm << std::endl;
      std::cout << "Norm of core tensor: " << std::setprecision(7) << gnorm << std::endl;
      std::string coreFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_core.mpi";
      std::cout << "Writing core tensor to " << coreFilename << std::endl;
      TuckerKokkos::exportTensorBinary(f.getG(), coreFilename.c_str());
    }

  } // local scope to ensure all Kokkos views are destructed appropriately
  Kokkos::finalize();

  return 0;
}
