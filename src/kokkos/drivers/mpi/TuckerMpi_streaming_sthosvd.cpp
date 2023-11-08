/*
 * driver.cpp
 *
 *  Created on: Nov 8, 2022
 *      Author: Hemanth Kolla (hnkolla@sandia.gov)
 */

#include "CmdLineParse.hpp"
#include "ParserInputParametersStreammingSthosvdDriver.hpp"
#include "TuckerMpi.hpp"
#include "TuckerMpi_StreamingTuckerTensor.hpp"
#include <cmath>
#include <iostream>

int main(int argc, char* argv[])
{
  using scalar_t = double;

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    int mpiRank, nprocs;
    const MPI_Comm& comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &mpiRank);
    MPI_Comm_size(comm, &nprocs);

    Tucker::Timer totalTimer;
    totalTimer.start();

    /*
     * read data and create tensor
     */
    const auto paramfn = Tucker::parse_cmdline_or(
      argc, (const char**)argv, "--parameter-file", "paramfile.txt");
    const InputParametersStreamingSthosvdDriver<scalar_t> inputs(
      paramfn, mpiRank);
    inputs.describe();
    Tucker::Timer readTimer;
    readTimer.start();
    const auto dataTensorDim = inputs.dimensionsOfDataTensor();
    TuckerMpi::Tensor<scalar_t, memory_space> X(inputs.dimensionsOfDataTensor(),
                                                inputs.proc_grid_dims);
    TuckerMpi::read_tensor_binary(mpiRank, X, inputs.rawDataFilenames);
    readTimer.stop();

    if(mpiRank == 0) {
      const size_t local_nnz = X.localSize();
      const size_t global_nnz = X.globalSize();
      std::cout << "Local initial tensor size  : ";
      Tucker::write_view_to_stream_singleline(std::cout, X.localDimensionsOnHost());
      std::cout << ", or "; Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(scalar_t));
      std::cout << "Global input tensor size : ";
      Tucker::write_view_to_stream_singleline(std::cout, X.globalDimensionsOnHost());
      std::cout << ", or "; Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(scalar_t));
    }
    MPI_Barrier(comm);

    /*
     * preprocessing
     */
    Tucker::Timer preprocessTimer;
    preprocessTimer.start();
    const int scaleMode = inputs.scale_mode;

    auto writeScalesShifts = [=](auto scales, auto shifts) {
      // Import scales and shifts to proc 0 in the column communicator
      // (so they are replicated across the row communicator)
      const auto& dist = X.getDistribution();
      const auto& grid = dist.getProcessorGrid();
      const MPI_Comm& col_comm = grid.getColComm(scaleMode, false);
      int col_mpi_rank = 0;
      MPI_Comm_rank(col_comm, &col_mpi_rank);
      using scales_type = decltype(scales);
      using shifts_type = decltype(scales);
      scales_type scales_all;
      shifts_type shifts_all;
      if (col_mpi_rank == 0) {
        scales_all = scales_type("scales_all", X.globalExtent(scaleMode));
        shifts_all = shifts_type("shifts_all", X.globalExtent(scaleMode));
      }
      const auto& recvcounts =
        dist.getMap(scaleMode, false)->getNumElementsPerProc();
      const auto& displs =
        dist.getMap(scaleMode, false)->getOffsets();
      TuckerMpi::MPI_Gatherv_(
        scales.data(), X.localExtent(scaleMode), scales_all.data(),
        recvcounts.data(), displs.data(), 0, col_comm);
      TuckerMpi::MPI_Gatherv_(
        shifts.data(), X.localExtent(scaleMode), shifts_all.data(),
        recvcounts.data(), displs.data(), 0, col_comm);

      // Copy to host
      auto scales_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scales_all);
      auto shifts_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shifts_all);

      // Write file using only (global) proc 0
      if (mpiRank == 0) {
        const std::string scale_file =
          inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_scale.txt";
        std::ofstream outStream(scale_file);
        outStream << scaleMode << std::endl;
        // Set output precision to match ScalarType representation (8 or 16)
        outStream << std::fixed
                  << std::setprecision(std::numeric_limits<scalar_t>::digits);
        for (std::size_t i=0; i<X.globalExtent(scaleMode); ++i){
          outStream << scales_h(i) << " " << shifts_h(i) << std::endl;
        }
        outStream.close();
      }
    };

    if(mpiRank == 0)
      std::cout << "Compute statistics" << std::endl;
    auto metricsData = TuckerMpi::compute_slice_metrics(mpiRank, X, scaleMode, Tucker::defaultMetrics);
    TuckerMpi::write_statistics(
      mpiRank, X.rank(), scaleMode, X.getDistribution(), metricsData,
      inputs.stats_file, inputs.stdThresh);

    Kokkos::View<scalar_t*, memory_space> scales;
    Kokkos::View<scalar_t*, memory_space> shifts;
    if (inputs.scaling_type != "None"){
      if(mpiRank == 0)
        std::cout << "Normalizing tensor" << std::endl;
      std::tie(scales, shifts) = TuckerMpi::normalize_tensor(
        mpiRank, X, metricsData, inputs.scaling_type,
        inputs.scale_mode, inputs.stdThresh);
      writeScalesShifts(scales, shifts);
    }
    else{
      if(mpiRank == 0)
        std::cout << "inputs.scaling_type == None, therefore we are not normalizing the tensor\n";
    }
    preprocessTimer.stop();
    if (inputs.boolWriteTensorAfterPreprocessing){
      TuckerMpi::write_tensor_binary(mpiRank, X, inputs.preprocDataFilenames);
    }

    /*
     * prepare lambdas "expressing" the computation to do
     */
    auto writeEigenvaluesToFiles = [=](auto container)
    {
      if (mpiRank == 0) {
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
      }
    };

    auto writeExtentsOfCoreTensor = [=](auto factorization)
    {
      if (mpiRank==0) {
        const std::string dimFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_ranks.txt";
        std::cout << "Writing core tensor extents to " << dimFilename << std::endl;
        std::ofstream of(dimFilename);
        assert(of.is_open());
        auto global_dims = factorization.coreTensor().globalDimensionsOnHost();
        for(int mode=0; mode<inputs.nd; mode++) {
          of << global_dims(mode) << std::endl;
        }
        of.close();
      }
    };

    auto writeExtentsOfGlobalTensor = [=](auto factorization)
    {
      if (mpiRank==0) {
        const std::string sizeFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_size.txt";
        std::cout << "Writing global tensor extents to " << sizeFilename << std::endl;
        std::ofstream of(sizeFilename);
        assert(of.is_open());
        for(int mode=0; mode<inputs.nd; mode++) {
          of << factorization.factorMatrix(mode).extent(0) << std::endl;
        }
        of.close();
      }
    };

    auto writeCoreTensorToFile = [=](auto factorization)
    {
      const std::string coreFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_core.mpi";
      if (mpiRank==0)
        std::cout << "Writing core tensor to " << coreFilename << std::endl;
      TuckerMpi::write_tensor_binary(mpiRank, factorization.coreTensor(), coreFilename);
    };

    auto writeEachFactor = [=](auto factorization)
    {
      if (mpiRank==0) {
        for(int mode=0; mode<inputs.nd; mode++) {
          // Create the filename by appending the mode #
          const std::string factorFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_mat_" + std::to_string(mode) + ".mpi";
          std::cout << "Writing factor " << mode << " to " << factorFilename << std::endl;
          Tucker::write_view_to_binary_file(factorization.factorMatrix(mode), factorFilename);
        }
      }
    };

    auto printNorms = [=](auto factorization){
      const auto xnorm = std::sqrt(X.frobeniusNormSquared());
      const auto gnorm = std::sqrt(factorization.coreTensor().frobeniusNormSquared());
      if (mpiRank==0) {
        std::cout << "Norm of input tensor: " << std::setprecision(7) << xnorm << std::endl;
        std::cout << "Norm of core tensor: " << std::setprecision(7) << gnorm << std::endl;
      }
    };

    Tucker::Timer sthosvdTimer, streamingSthosvdTimer, streamingReadTimer, writeTimer;
    if(inputs.boolSTHOSVD) {
      /////////////////////////////
      // Perform Initial STHOSVD //
      /////////////////////////////
      auto truncator =
        Tucker::create_core_tensor_truncator(X, inputs.dimensionsOfCoreTensor(), inputs.tol, mpiRank);

      const auto method = TuckerMpi::Method::NewGram;
      sthosvdTimer.start();
      auto [initial_solution, initial_eigvals] = TuckerMpi::sthosvd(
        method, X, truncator, inputs.modeOrder, false /*flipSign*/);
      sthosvdTimer.stop();

      //std::cout<< "\n";
      writeEigenvaluesToFiles(initial_eigvals);

      /////////////////////////////
      // Perform Streaming HOSVD //
      /////////////////////////////
      streamingSthosvdTimer.start();
      TuckerMpi::StreamingTuckerTensor<scalar_t> solution =
        TuckerMpi::StreamingSTHOSVD(
          X, initial_solution, initial_eigvals,
          inputs.scale_mode, scales, shifts,
          inputs.streaming_fns_file.c_str(),
          inputs.tol, streamingReadTimer, inputs.streaming_stats_file);
      streamingSthosvdTimer.stop();

      /////////////////////////
      // Compute Error Bound //
      /////////////////////////

      const scalar_t xnorm = std::sqrt(solution.Xnorm2);
      if (mpiRank==0)
        std::cout << "Norm of input tensor: " << xnorm << std::endl;

      const scalar_t gnorm = std::sqrt(solution.factorization.coreTensor().frobeniusNormSquared());
      if (mpiRank==0)
        std::cout << "Norm of core tensor: " << gnorm << std::endl;

      scalar_t enorm = 0.0;
      for(int i = 0; i < inputs.nd; ++i) {
        enorm += solution.squared_errors[i];
      }
      enorm = std::sqrt(enorm);
      if (mpiRank==0) {
        std::cout << "Error bound: absolute = " << enorm << ", relative = " << enorm / xnorm << std::endl;

        std::cout << "Core tensor size after StreamingSTHOSVD iterations: " << std::flush;
        Tucker::write_view_to_stream_singleline(std::cout, solution.factorization.coreTensor().globalDimensionsOnHost());
        std::cout << std::endl;
      }

      writeTimer.start();
      if(inputs.boolWriteResultsOfSTHOSVD) {
        writeExtentsOfCoreTensor(solution.factorization);
        writeExtentsOfGlobalTensor(solution.factorization);
        writeCoreTensorToFile(solution.factorization);
        writeEachFactor(solution.factorization);
      }
      writeTimer.stop();
    }

    TuckerMpi::print_max_mem_usage_to_stream(comm, std::cout);

    totalTimer.stop();

    double read_time_l = readTimer.duration();
    double preprocess_time_l = preprocessTimer.duration();
    double sthosvd_time_l = sthosvdTimer.duration();
    double streaming_read_time_l = streamingReadTimer.duration();
    double streaming_sthosvd_time_l = streamingSthosvdTimer.duration() - streamingReadTimer.duration();
    double write_time_l = writeTimer.duration();
    double total_time_l = totalTimer.duration();

    double read_time = 0;
    double preprocess_time = 0;
    double sthosvd_time = 0;
    double streaming_read_time = 0;
    double streaming_sthosvd_time = 0;
    double write_time = 0;
    double total_time = 0;

    TuckerMpi::MPI_Reduce_(&read_time_l,&read_time,1,MPI_MAX,0,comm);
    TuckerMpi::MPI_Reduce_(&preprocess_time_l,&preprocess_time,1,MPI_MAX,0,comm);
    TuckerMpi::MPI_Reduce_(&sthosvd_time_l,&sthosvd_time,1,MPI_MAX,0,comm);
    TuckerMpi::MPI_Reduce_(&streaming_read_time_l,&streaming_read_time,1,MPI_MAX,0,comm);
    TuckerMpi::MPI_Reduce_(&streaming_sthosvd_time_l,&streaming_sthosvd_time,1,MPI_MAX,0,comm);
    TuckerMpi::MPI_Reduce_(&write_time_l,&write_time,1,MPI_MAX,0,comm);
    TuckerMpi::MPI_Reduce_(&total_time_l,&total_time,1,MPI_MAX,0,comm);

    if (mpiRank==0) {
      std::cout << "Initial read time: " << read_time << std::endl;
      std::cout << "Preprocessing time: " << preprocess_time << std::endl;
      std::cout << "Initial STHOSVD time: " << sthosvd_time << std::endl;
      std::cout << "Streaming read time: " << streaming_read_time << std::endl;
      std::cout << "Streaming STHOSVD time: " << streaming_sthosvd_time << std::endl;
      std::cout << "Write time: " << write_time << std::endl;
      std::cout << "Total time: " << total_time << std::endl;
    }
  } // local scope to ensure all Kokkos views are destructed appropriately
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
