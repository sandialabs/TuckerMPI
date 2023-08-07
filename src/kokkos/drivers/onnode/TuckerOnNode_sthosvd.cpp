
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
    TuckerOnNode::read_tensor_binary(X, inputs.rawDataFilenames);

    /*
     * preprocessing
     */
    const int scaleMode = inputs.scale_mode;

    auto writeScalesShifts = [=](auto scales, auto shifts){
      auto scales_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scales);
      auto shifts_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shifts);
      const std::string scale_file = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_scale.txt";
      std::ofstream outStream(scale_file);
      outStream << scaleMode << std::endl;
      // Set output precision to match ScalarType representation (8 or 16)
      outStream << std::fixed << std::setprecision(std::numeric_limits<scalar_t>::digits);
      for(int i=0; i<X.extent(scaleMode); i++){
        outStream << scales_h(i) << " " << shifts_h(i) << std::endl;
      }
      outStream.close();
    };

    std::cout << "Compute statistics" << std::endl;
    const std::vector<Tucker::Metric> metrics{Tucker::Metric::MIN,  Tucker::Metric::MAX,
					      Tucker::Metric::MEAN, Tucker::Metric::VARIANCE};
    auto metricsData = TuckerOnNode::compute_slice_metrics(X, scaleMode, metrics);
    Tucker::write_statistics(metricsData, inputs.stats_file, inputs.stdThresh);

    if (inputs.scaling_type != "None"){
      std::cout << "Normalize tensor if needed" << std::endl;
      auto [scales, shifts] = TuckerOnNode::normalize_tensor(X, inputs.scaling_type, inputs.scale_mode, inputs.stdThresh);
      writeScalesShifts(scales, shifts);
    }
    else{
      std::cout << "inputs.scaling_type == None, therefore we are not normalizing the tensor\n";
    }
    if (inputs.boolWriteTensorAfterPreprocessing){
      TuckerOnNode::write_tensor_binary(X, inputs.pre_fns_file);
    }

    /*
     * prepare lambdas "expressing" the computation to do
     */
    auto writeEigenvaluesToFiles = [=](auto container)
    {
      const std::string filePrefix = inputs.sv_dir + "/" + inputs.sv_fn + "_mode_";
      // FIXME: need to figure out: https://gitlab.com/nga-tucker/TuckerMPI/-/issues/15

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
    };

    auto writeExtentsOfCoreTensor = [=](auto factorization)
    {
      const std::string dimFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_ranks.txt";
      std::cout << "Writing core tensor extents to " << dimFilename << std::endl;
      std::ofstream of(dimFilename);
      assert(of.is_open());
      for(int mode=0; mode<inputs.nd; mode++) {
        of << factorization.coreTensor().extent(mode) << std::endl;
      }
      of.close();
    };

    auto writeExtentsOfGlobalTensor = [=]()
    {
      const std::string sizeFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_size.txt";
      std::cout << "Writing global tensor extents to " << sizeFilename << std::endl;
      std::ofstream of(sizeFilename);
      assert(of.is_open());
      for(int mode=0; mode<inputs.nd; mode++) {
        of << inputs.dimensionsOfDataTensor()[mode] << std::endl;
      }
      of.close();
    };

    auto writeCoreTensorToFile = [=](auto factorization)
    {
      const std::string coreFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_core.mpi";
      std::cout << "Writing core tensor to " << coreFilename << std::endl;
      TuckerOnNode::write_tensor_binary(factorization.coreTensor(), coreFilename);
    };

    auto writeEachFactor = [=](auto factorization)
    {
      for(int mode=0; mode<inputs.nd; mode++) {
        // Create the filename by appending the mode #
        const std::string factorFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_mat_" + std::to_string(mode) + ".mpi";
        std::cout << "Writing factor " << mode << " to " << factorFilename << std::endl;
        Tucker::write_view_to_binary_file(factorization.factorMatrix(mode), factorFilename);
      }
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

      std::cout<< "\n";
      writeEigenvaluesToFiles(eigvals);

      printNorms(tt);

      // FIXME: Compute the error bound based on the eigenvalues

      if(inputs.boolWriteResultsOfSTHOSVD){
        writeExtentsOfCoreTensor(tt);
        writeExtentsOfGlobalTensor();
        writeCoreTensorToFile(tt);
        writeEachFactor(tt);
      }
    };

    /* run for real */
    if(inputs.boolSTHOSVD){
      sthosvdGram(truncator);
    }

  } // local scope to ensure all Kokkos views are destructed appropriately
  Kokkos::finalize();

  return 0;
}
