#include "CmdLineParse.hpp"
#include "ParserInputParametersGenerateDriver.hpp"
#include "TuckerOnNode.hpp"
#include <random>

int main(int argc, char* argv[])
{
  using scalar_t = double;

  Kokkos::initialize(argc, argv);
  {
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

    //
    // read paramfile
    const auto paramfn = Tucker::parse_cmdline_or(argc, (const char**)argv,
						  "--parameter-file", "paramfile.txt");
    const InputParametersGenerateDriver<scalar_t> inputs(paramfn);
    inputs.describe();

    //
    // get params
    const auto seed = inputs.seed_;
    const auto eps  = inputs.epsilon_;
    const auto I_dims = inputs.dataTensorDims_;
    const auto R_dims = inputs.coreTensorDims_;
    const int tensorRank = I_dims.size();

    //
    // aliases
    using tensor_type   = TuckerOnNode::Tensor<scalar_t, memory_space>;
    using factor_view_t = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space>;

    //
    // run
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution;

    // core tensor
    std::cout << "Generating a random core tensor...\n";
    tensor_type G(R_dims);
    auto G_h = Tucker::create_mirror_tensor_and_copy(Kokkos::HostSpace(), G);
    auto G_view_h = G_h.data();
    for(size_t i=0; i<G.size(); i++) {
      G_view_h(i) = distribution(generator);
    }
    Tucker::deep_copy(G, G_h);
    const std::size_t nnz = G.size();

    // factor matrices and ttm
    tensor_type Y = G;
    for(int d=0; d<tensorRank; d++)
    {
      std::cout << "Generating factor matrix " << d << "...\n";
      const int nrows = I_dims[d];
      const int ncols = R_dims[d];
      auto M = factor_view_t("currentFactorMatrix", nrows, ncols);
      auto M_h = Kokkos::create_mirror(M);
      for(size_t j=0; j<M.extent(1); j++) {
	for(size_t i=0; i<M.extent(0); i++) {
	  M_h(i,j) = distribution(generator);
	}
      }
      Kokkos::deep_copy(M, M_h);

      std::cout << "Performing mode " << d << " TTM...\n";
      auto temp = TuckerOnNode::ttm(Y, d, M, false);
      Y = temp;
    }

    // Compute the norm of the global tensor
    std::cout << "Computing the global tensor norm...\n";
    const auto normM = std::sqrt(Y.frobeniusNormSquared());

    ///////////////////////////////////////////////////////////////////
    // Compute the estimated norm of the noise matrix
    // The average of each element squared is the standard deviation
    // squared, so this quantity should be sqrt(nnz * stdev^2)
    ///////////////////////////////////////////////////////////////////
    const scalar_t normN = std::sqrt((scalar_t) nnz);
    const scalar_t alpha = eps*normM/normN;

    //////////////////////////////////////////////////////////////////////
    // For each entry of the global tensor, add alpha*randn
    // Note that this is space-efficient, as we do not store the entire noise tensor
    //////////////////////////////////////////////////////////////////////
    std::cout << "Adding noise...\n";
    auto Y_h = Tucker::create_mirror_tensor_and_copy(Kokkos::HostSpace(), Y);
    auto Y_h_view = Y_h.data();
    for(size_t i=0; i<nnz; i++) {
      Y_h_view(i) += alpha*distribution(generator);
    }
    Tucker::deep_copy(Y, Y_h);

    TuckerOnNode::write_tensor_binary(Y, inputs.outDataFilenames);

  } // local scope to ensure all Kokkos views are destructed appropriately
  Kokkos::finalize();
  return 0;
}
