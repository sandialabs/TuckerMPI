#include "CmdLineParse.hpp"
#include "ParserInputParametersGenerateDriver.hpp"
#include "TuckerMpi.hpp"

auto generate_seed(const int mpiRank, const int nprocs, unsigned seedIn)
{
  int myseed;
  if(mpiRank == 0) {
    std::vector<unsigned int> seeds(nprocs);
    srand(seedIn);
    for(int i=0; i<nprocs; i++) { seeds[i] = rand(); }
    MPI_Scatter(seeds.data(), 1, MPI_INT,&myseed,1,MPI_INT,0,MPI_COMM_WORLD);
  }
  else {
    MPI_Scatter(NULL,1,MPI_INT,&myseed,1,MPI_INT,0,MPI_COMM_WORLD);
  }

  return (unsigned int) myseed;
}

int main(int argc, char* argv[])
{
  using scalar_t = double;

  MPI_Init(&argc, &argv);

  int mpiRank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  Kokkos::initialize(argc, argv);
  {
    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
						  "--parameter-file", "paramfile.txt");
    const InputParametersGenerateDriver<scalar_t> inputs(paramfn, mpiRank);
    if (mpiRank==0){
      inputs.describe();
    }

    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

    //
    // get params
    const auto eps  = inputs.epsilon_;
    const auto procGrid = inputs.procGridDims_;
    const auto I_dims = inputs.dataTensorDims_;
    const auto R_dims = inputs.coreTensorDims_;
    const int tensorRank = I_dims.size();

    //
    // aliases
    using tensor_type = TuckerMpi::Tensor<scalar_t, memory_space>;
    using factor_view_t = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space>;

    // Generate the seeds for each MPI process
    const auto myseed = generate_seed(mpiRank, nprocs, inputs.seed_);


    //
    // run
    std::default_random_engine generator(myseed);
    std::normal_distribution<scalar_t> distribution;

    TuckerMpi::Distribution procDist(R_dims, procGrid);

    // core tensor
    if (mpiRank==0){
      std::cout << "Generating a random core tensor...\n";
    }
    tensor_type G(procDist);

    auto Gl = G.localTensor();
    auto Gl_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), Gl);
    auto Gl_view_h = Gl_h.data();
    for(size_t i=0; i<Gl_h.size(); i++) {
      Gl_view_h(i) = distribution(generator);
    }
    Tucker::deep_copy(Gl, Gl_h);

    // factor matrices and ttm
    tensor_type Y = G;
    for(int d=0; d<tensorRank; d++)
    {
      if (mpiRank==0) std::cout << "Generating factor matrix " << d << "...\n";
      const int nrows = I_dims[d];
      const int ncols = R_dims[d];
      auto M = factor_view_t("currentFactorMatrix", nrows, ncols);
      auto M_h = Kokkos::create_mirror(M);
      if(mpiRank == 0) {
      	for(size_t j=0; j<M.extent(1); j++) {
      	  for(size_t i=0; i<M.extent(0); i++) {
      	    M_h(i,j) = distribution(generator);
      	  }
      	}
      }
      TuckerMpi::MPI_Bcast_(M_h.data(), M_h.size(), 0, MPI_COMM_WORLD);
      Kokkos::deep_copy(M, M_h);

      if (mpiRank==0){ std::cout << "Performing mode " << d << " TTM...\n"; }
      auto temp = TuckerMpi::ttm(Y, d, M, false, 0);

      // need to do = {} first, otherwise Y=temp throws because Y = temp
      // is assigning tensors with different distributions
      Y = {};
      Y = temp;
    }

    // Compute the norm of the global tensor
    if (mpiRank==0){ std::cout << "Computing the global tensor norm...\n"; }
    const auto normM = std::sqrt(Y.frobeniusNormSquared());
    if (mpiRank==0){ std::cout << "global tensor norm = " << normM << "\n"; }

    ///////////////////////////////////////////////////////////////////
    // Compute the estimated norm of the noise matrix
    // The average of each element squared is the standard deviation
    // squared, so this quantity should be sqrt(nnz * stdev^2)
    ///////////////////////////////////////////////////////////////////
    const std::size_t init = 1;
    const std::size_t nnz = std::accumulate(std::cbegin(I_dims), std::cend(I_dims), init, std::multiplies<std::size_t>());
    const scalar_t normN = std::sqrt((scalar_t) nnz);
    const scalar_t alpha = eps*normM/normN;

    //////////////////////////////////////////////////////////////////////
    // For each entry of the global tensor, add alpha*randn
    // Note that this is space-efficient, as we do not store the entire noise tensor
    //////////////////////////////////////////////////////////////////////
    if (mpiRank==0){ std::cout << "Adding noise...\n"; }
    auto Y_local = Y.localTensor();
    auto Yl_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), Y_local);
    auto Y_h_view = Yl_h.data();
    for(size_t i=0; i<Y_h_view.extent(0); i++) {
      Y_h_view(i) += alpha*distribution(generator);
    }
    Tucker::deep_copy(Y_local, Yl_h);

    if(mpiRank == 0) std::cout << "Writing tensor to disk...\n";
    TuckerMpi::write_tensor_binary(mpiRank, Y, inputs.outDataFilenames);

  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
