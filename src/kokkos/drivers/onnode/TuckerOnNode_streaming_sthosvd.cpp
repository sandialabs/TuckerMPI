/*
 * driver.cpp
 *
 *  Created on: Nov 8, 2022
 *      Author: Hemanth Kolla (hnkolla@sandia.gov)
 */

#include "CmdLineParse.hpp"
#include "ParserInputParametersStreammingSthosvdDriver.hpp"
#include "TuckerOnNode.hpp"
#include "TuckerOnNode_StreamingTuckerTensor.hpp"
#include <cmath>
#include <iostream>

int main(int argc, char* argv[])
{
  using scalar_t = double;

  Kokkos::initialize(argc, argv);
  {
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

    Tucker::Timer totalTimer;
    totalTimer.start();

    /*
     * read data and create tensor
     */
    const auto paramfn = Tucker::parse_cmdline_or(
      argc, (const char**)argv, "--parameter-file", "paramfile.txt");
    const InputParametersStreamingSthosvdDriver<scalar_t> inputs(paramfn);
    inputs.describe();
    Tucker::Timer readTimer;
    readTimer.start();
    TuckerOnNode::Tensor<scalar_t, memory_space> X(inputs.dimensionsOfDataTensor());
    TuckerOnNode::read_tensor_binary(X, inputs.rawDataFilenames);
    readTimer.stop();

    size_t nnz = X.size();
    std::cout << "Initial input tensor size: ";
    for (int i=0; i<X.rank(); ++i)
      std::cout << X.extent(i) << " ";
    std::cout << ", or ";
    Tucker::print_bytes_to_stream(std::cout, nnz*sizeof(double));

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
      for(std::size_t i=0; i<X.extent(scaleMode); i++){
        outStream << scales_h(i) << " " << shifts_h(i) << std::endl;
      }
      outStream.close();
    };

    std::cout << "Compute statistics" << std::endl;
    auto metricsData = TuckerOnNode::compute_slice_metrics(X, scaleMode, Tucker::defaultMetrics);
    TuckerOnNode::write_statistics(metricsData, inputs.stats_file, inputs.stdThresh);

    if (inputs.scaling_type != "None"){
      std::cout << "Normalizing tensor" << std::endl;
      auto [scales, shifts] = TuckerOnNode::normalize_tensor(X, metricsData, inputs.scaling_type,
                                                             inputs.scale_mode, inputs.stdThresh);
      writeScalesShifts(scales, shifts);
    }
    else{
      std::cout << "inputs.scaling_type == None, therefore we are not normalizing the tensor\n";
    }
    if (inputs.boolWriteTensorAfterPreprocessing){
      TuckerOnNode::write_tensor_binary(X, inputs.preprocDataFilenames);
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

        auto eigvals = container[mode];
        auto eigvals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigvals);
        for(std::size_t i=0; i<eigvals.extent(0); i++) {
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

    auto writeExtentsOfGlobalTensor = [=](auto factorization)
    {
      const std::string sizeFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_size.txt";
      std::cout << "Writing global tensor extents to " << sizeFilename << std::endl;
      std::ofstream of(sizeFilename);
      assert(of.is_open());
      for(int mode=0; mode<inputs.nd; mode++) {
        of << factorization.factorMatrix(mode).extent(0) << std::endl;
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

    Tucker::Timer sthosvdTimer, streamingSthosvdTimer, streamingReadTimer, writeTimer;
    if(inputs.boolSTHOSVD) {
      /////////////////////////////
      // Perform Initial STHOSVD //
      /////////////////////////////
      auto truncator =
        Tucker::create_core_tensor_truncator(X, inputs.dimensionsOfCoreTensor(), inputs.tol);

      const auto method = TuckerOnNode::Method::Gram;
      sthosvdTimer.start();
      auto [initial_solution, initial_eigvals] = TuckerOnNode::sthosvd(method, X, truncator, false /*flipSign*/);
      sthosvdTimer.stop();

      std::cout<< "\n";
      writeEigenvaluesToFiles(initial_eigvals);

      /////////////////////////////
      // Perform Streaming HOSVD //
      /////////////////////////////
      streamingSthosvdTimer.start();
      TuckerOnNode::StreamingTuckerTensor<scalar_t> solution =
        TuckerOnNode::StreamingSTHOSVD(
          X, initial_solution, initial_eigvals,
          inputs.streaming_fns_file.c_str(),
          inputs.tol, streamingReadTimer, inputs.streaming_stats_file);
      streamingSthosvdTimer.stop();

      /////////////////////////
      // Compute Error Bound //
      /////////////////////////

      const scalar_t xnorm = std::sqrt(solution.Xnorm2);
      std::cout << "Norm of input tensor: " << xnorm << std::endl;

      const scalar_t gnorm = std::sqrt(solution.factorization.coreTensor().frobeniusNormSquared());
      std::cout << "Norm of core tensor: " << gnorm << std::endl;

      scalar_t enorm = 0.0;
      for(int i = 0; i < inputs.nd; ++i) {
        enorm += solution.squared_errors[i];
      }
      enorm = std::sqrt(enorm);
      std::cout << "Error bound: absolute = " << enorm << ", relative = " << enorm / xnorm << std::endl;

      std::cout << "Local tensor size after StreamingSTHOSVD iterations:" << std::flush;
      for (int i = 0; i < inputs.nd; ++i) {
        std::cout << " " << solution.factorization.coreTensor().extent(i) << std::flush;
      }
      std::cout << std::endl;

      writeTimer.start();
      if(inputs.boolWriteResultsOfSTHOSVD) {
        writeExtentsOfCoreTensor(solution.factorization);
        writeExtentsOfGlobalTensor(solution.factorization);
        writeCoreTensorToFile(solution.factorization);
        writeEachFactor(solution.factorization);
      }
      writeTimer.stop();
    }

    Tucker::print_max_mem_usage_to_stream(std::cout);

    totalTimer.stop();
    std::cout << "Initial read time: " << readTimer.duration() << std::endl;
    std::cout << "Initial STHOSVD time: " << sthosvdTimer.duration() << std::endl;
    std::cout << "Streaming read time: " << streamingReadTimer.duration() << std::endl;
    std::cout << "Streaming STHOSVD time: " << streamingSthosvdTimer.duration() - streamingReadTimer.duration() << std::endl;
    std::cout << "Write time: " << writeTimer.duration() << std::endl;
    std::cout << "Total time: " << totalTimer.duration() << std::endl;

  } // local scope to ensure all Kokkos views are destructed appropriately
  Kokkos::finalize();

  return 0;
}