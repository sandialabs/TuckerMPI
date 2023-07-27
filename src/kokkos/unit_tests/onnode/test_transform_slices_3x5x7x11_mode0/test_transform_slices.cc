/**
 * Originally import test here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/serial/tests/Tucker_shift_scale_test.cpp
 */

#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode0){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 3x5x7x11 tensor
  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);

  // Read tensor from file
  TuckerOnNode::import_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  // Define scales and shifts
  Kokkos::View<scalar_t*, memory_space> scales("scales", 3);
  scales(0) = -0.258915944830978;
  scales(1) =  0.341369101972144;
  scales(2) =  0.357212764090606;

  Kokkos::View<scalar_t*, memory_space> shifts("shifts", 3);
  shifts(0) = -0.450368098480610;
  shifts(1) = -0.408899824586946;
  shifts(2) =  0.094037031444121;

  // Read true solution from file
  TuckerOnNode::Tensor<scalar_t, memory_space> true_sol(dims);
  TuckerOnNode::import_tensor_binary(true_sol, "./gold_3x5x7x11_ss0.bin");

  // Call shift-scale
  int mode = 0;
  Tucker::transform_slices(tensor, mode, scales, shifts);

  // Checks
  // TODO: Create functions checks(t1, t2, tol)
  // TODO: with impl checks(tensor, true_sol, 100 * std::numeric_limits<scalar_t>::epsilon());
  // 1) Owns any data
  ASSERT_NE(tensor.size(), 0);
  ASSERT_NE(true_sol.size(), 0);

  // 2) Same size
  ASSERT_EQ(tensor.size(), true_sol.size());

  // 3) Values
  int numElements = tensor.size();
  scalar_t errNorm2 = 0;
  for(int i=0; i<numElements; i++) {
    scalar_t err = std::abs(tensor.data()[i] - true_sol.data()[i]);
    ASSERT_FALSE(std::isnan(err));
    errNorm2 += (err*err);
  }

  // 4) Tol
  scalar_t origNorm2 = tensor.frobeniusNormSquared();
  scalar_t relErr = std::sqrt(errNorm2/origNorm2);
  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_FALSE(relErr > tol);
}

// Add tests
