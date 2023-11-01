#include "CmdLineParse.hpp"
#include "ParserInputParametersSthosvdDriver.hpp"
#include "TuckerMpi.hpp"

template<class ScalarType>
void run(const InputParametersSthosvdDriver<ScalarType> & inputs)
{
  // ------------------------------------------------------
  // prepare
  // ------------------------------------------------------
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  int mpiRank, nprocs;
  const MPI_Comm& comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &mpiRank);
  MPI_Comm_size(comm, &nprocs);

  Tucker::Timer totalTimer;
  totalTimer.start();

  // ------------------------------------------------------
  // read data and create tensor
  // ------------------------------------------------------
  if(mpiRank == 0) { inputs.describe(); }

  const auto dataTensorDim = inputs.dimensionsOfDataTensor();
  Tucker::Timer readTimer;
  readTimer.start();
  TuckerMpi::Tensor<ScalarType, memory_space> X(dataTensorDim, inputs.proc_grid_dims);
  TuckerMpi::read_tensor_binary(mpiRank, X, inputs.rawDataFilenames);
  readTimer.stop();

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
  MPI_Barrier(comm);

  // ------------------------------------------------------
  // preprocessing
  // ------------------------------------------------------
  Tucker::Timer preprocessTimer;
  preprocessTimer.start();
  const int scaleMode = inputs.scale_mode;
  if(mpiRank == 0) {
    std::cout << "Compute statistics" << std::endl;
  }
  auto metricsData = TuckerMpi::compute_slice_metrics(mpiRank, X, scaleMode, Tucker::defaultMetrics);
  TuckerMpi::write_statistics(mpiRank, X.rank(), scaleMode, X.getDistribution(),
                              metricsData, inputs.stats_file, inputs.stdThresh);

  if (inputs.scaling_type != "None"){
    if(mpiRank == 0) {
      std::cout << "Normalizing tensor" << std::endl;
    }
    auto [scales, shifts] = TuckerMpi::normalize_tensor(
      mpiRank, X, metricsData, inputs.scaling_type,
      inputs.scale_mode, inputs.stdThresh);
    //TuckerwriteScalesShifts(scales, shifts);
  }
  else{
    if(mpiRank == 0) {
      std::cout << "inputs.scaling_type == None, therefore we are not normalizing the tensor\n";
    }
  }
  preprocessTimer.stop();
  if (inputs.boolWriteTensorAfterPreprocessing){
    TuckerMpi::write_tensor_binary(mpiRank, X, inputs.preprocDataFilenames);
  }

  // ------------------------------------------------------
  // prepare bricks "expressing" the computation to do
  // ------------------------------------------------------
  auto writeEigenvaluesToFile = [=](auto container){
    if(mpiRank == 0) {
      const std::string filePrefix = inputs.sv_dir + "/" + inputs.sv_fn + "_mode_";

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

  auto printNorms = [=](auto factorization){
    auto xnorm2 = X.frobeniusNormSquared();
    auto xnorm  = std::sqrt(xnorm2);
    auto gnorm  = std::sqrt(factorization.coreTensor().frobeniusNormSquared());
    if(mpiRank == 0) {
      std::cout << "Norm of input tensor: " << std::setprecision(15) << xnorm << std::endl;
      std::cout << "Norm of core tensor: "  << std::setprecision(15) << gnorm << std::endl;
    }
  };

  auto writeCoreTensorToFile = [=](auto factorization)
  {
    const std::string coreFilename = inputs.sthosvd_dir + "/" + inputs.sthosvd_fn + "_core.mpi";
    if (mpiRank==0){
      std::cout << "Writing core tensor to " << coreFilename << std::endl;
    }
    TuckerMpi::write_tensor_binary(mpiRank, factorization.coreTensor(), coreFilename);
  };

  auto writeEachFactor = [=](auto factorization)
  {
    for(int mode=0; mode<inputs.nd; mode++) {
      const std::string factorFilename = inputs.sthosvd_dir + "/" +
	inputs.sthosvd_fn + "_mat_" + std::to_string(mode) + ".mpi";
      std::cout << "Writing factor " << mode << " to " << factorFilename << std::endl;
      Tucker::write_view_to_binary_file(factorization.factorMatrix(mode), factorFilename);
    }
  };

  auto truncator =
    Tucker::create_core_tensor_truncator(X, inputs.dimensionsOfCoreTensor(), inputs.tol, mpiRank);

  Tucker::Timer sthosvdTimer, writeTimer;
  auto sthosvdNewGram = [&](auto truncator){
    sthosvdTimer.start();
    const auto method = TuckerMpi::Method::NewGram;
    auto [tt, eigvals] = TuckerMpi::sthosvd(
      method, X, truncator, inputs.modeOrder, false /*flipSign*/);
    sthosvdTimer.stop();
    writeEigenvaluesToFile(eigvals);
    printNorms(tt);

    writeTimer.start();
    if(inputs.boolWriteResultsOfSTHOSVD){
      writeCoreTensorToFile(tt);
      if (mpiRank==0){
        writeEachFactor(tt);
      }
    }
    writeTimer.stop();
  };

  // ------------------------------------------------------
  // use bricks and run for real
  // ------------------------------------------------------
  if(inputs.boolSTHOSVD){
    sthosvdNewGram(truncator);
  }

  TuckerMpi::print_max_mem_usage_to_stream(comm, std::cout);

  totalTimer.stop();
  double read_time_l = readTimer.duration();
  double preprocess_time_l = preprocessTimer.duration();
  double sthosvd_time_l = sthosvdTimer.duration();
  double write_time_l = writeTimer.duration();
  double total_time_l = totalTimer.duration();

  double read_time = 0;
  double preprocess_time = 0;
  double sthosvd_time = 0;
  double write_time = 0;
  double total_time = 0;

  TuckerMpi::MPI_Reduce_(&read_time_l,&read_time,1,MPI_MAX,0,comm);
  TuckerMpi::MPI_Reduce_(&preprocess_time_l,&preprocess_time,1,MPI_MAX,0,comm);
  TuckerMpi::MPI_Reduce_(&sthosvd_time_l,&sthosvd_time,1,MPI_MAX,0,comm);
  TuckerMpi::MPI_Reduce_(&write_time_l,&write_time,1,MPI_MAX,0,comm);
  TuckerMpi::MPI_Reduce_(&total_time_l,&total_time,1,MPI_MAX,0,comm);

  if (mpiRank==0) {
    std::cout << "Read time: " << read_time << std::endl;
    std::cout << "Preprocessing time: " << preprocess_time << std::endl;
    std::cout << "STHOSVD time: " << sthosvd_time << std::endl;
    std::cout << "Write time: " << write_time << std::endl;
    std::cout << "Total time: " << total_time << std::endl;
  }
}

int main(int argc, char* argv[])
{
  using scalar_t = double;

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    const auto paramfn = Tucker::parse_cmdline_or(
      argc, (const char**)argv, "--parameter-file", "paramfile.txt");
    const InputParametersSthosvdDriver<scalar_t> inputs(paramfn);
    run(inputs);
  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
