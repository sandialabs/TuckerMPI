/**
 * Originally import test here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/serial/tests/Tucker_shift_scale_test.cpp
 */

#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

template<class ScalarType, class MemorySpaces>
bool checks(TuckerOnNode::Tensor<ScalarType, MemorySpaces> t1,
            TuckerOnNode::Tensor<ScalarType, MemorySpaces> t2,
            ScalarType tol)
{
  // 1) Owns any data
  if(t1.size() == 0 && t2.size() == 0){
    return true;
  }

  // 2) Same size
  if(t1.size() != t2.size()) {
    return false;
  }

  // 3) Values
  int numElements = t1.size();
  ScalarType errNorm2 = 0;
  for(int i=0; i<numElements; i++) {
    ScalarType err = std::abs(t1.data()[i] - t2.data()[i]);
    if(std::isnan(err)){
      return false;
    }
    errNorm2 += (err*err);
  }

  // 4) Tol
  ScalarType origNorm2 = t1.frobeniusNormSquared();
  ScalarType relErr = std::sqrt(errNorm2/origNorm2);
  if(relErr > tol){
    return false;
  }

  // Return
  return true;
}

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
  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol, tol));
}

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode1){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 3x5x7x11 tensor
  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);

  // Read tensor from file
  TuckerOnNode::import_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  // Define scales and shifts
  Kokkos::View<scalar_t*, memory_space> scales("scales", 5);
  scales(0) =  0.262109709211147;
  scales(1) = -0.152432849551241;
  scales(2) = -0.038768240608501;
  scales(3) =  0.139323762199356;
  scales(4) =  0.417336040866845;

  Kokkos::View<scalar_t*, memory_space> shifts("shifts", 5);
  shifts(0) =  0.463612200951355;
  shifts(1) = -0.011100213839396;
  shifts(2) = -0.279689899431367;
  shifts(3) = -0.273791359158714;
  shifts(4) =  0.036787804512826;

  // Read true solution from file
  TuckerOnNode::Tensor<scalar_t, memory_space> true_sol(dims);
  TuckerOnNode::import_tensor_binary(true_sol, "./gold_3x5x7x11_ss1.bin");

  // Call shift-scale
  int mode = 1;
  Tucker::transform_slices(tensor, mode, scales, shifts);

  // Checks
  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol, tol));
}

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode2){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 3x5x7x11 tensor
  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);

  // Read tensor from file
  TuckerOnNode::import_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  // Define scales and shifts
  Kokkos::View<scalar_t*, memory_space> scales("scales", 7);
  scales(0) =  0.133333580320122; scales(1) =  0.124000554312344;
  scales(2) = -0.172058403026751; scales(3) =  0.302965315958237;
  scales(4) =  0.499477858635892; scales(5) =  0.480978160932146;
  scales(6) = -0.372963057805143;

  Kokkos::View<scalar_t*, memory_space> shifts("shifts", 7);
  shifts(0) = -0.338427426109669; shifts(1) =  0.215635404167474;
  shifts(2) =  0.077738876192409; shifts(3) = -0.066701022790881;
  shifts(4) =  0.384242782631094; shifts(5) = -0.106948244623087;
  shifts(6) = -0.321024847372268;

  // Read true solution from file
  TuckerOnNode::Tensor<scalar_t, memory_space> true_sol(dims);
  TuckerOnNode::import_tensor_binary(true_sol, "./gold_3x5x7x11_ss2.bin");

  // Call shift-scale
  int mode = 2;
  Tucker::transform_slices(tensor, mode, scales, shifts);

  // Checks
  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol, tol));
}

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode3){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 3x5x7x11 tensor
  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t, memory_space> tensor(dims);

  // Read tensor from file
  TuckerOnNode::import_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  // Define scales and shifts
  Kokkos::View<scalar_t*, memory_space> scales("scales", 11);
  scales(0) = -0.375974083351957; scales(1) = -0.029236754133043;
  scales(2) =  0.356896327782193; scales(3) = -0.456609528433047;
  scales(4) =  0.191625145201306; scales(5) =  0.478985466675038;
  scales(6) = -0.216732101507863; scales(7) = -0.366219500005577;
  scales(8) =  0.185279684412687; scales(9) =  0.409454555749395;
  scales(10) = 0.110868982383243;

  Kokkos::View<scalar_t*, memory_space> shifts("shifts", 11);
  shifts(0) = -0.267759854038207; shifts(1) = -0.476367533341775;
  shifts(2) =  0.107432610401855; shifts(3) = -0.389190678712850;
  shifts(4) = -0.092540492121601; shifts(5) =  0.384076806661962;
  shifts(6) =  0.048132777476588; shifts(7) = -0.130996923288383;
  shifts(8) = -0.291654017186653; shifts(9) = -0.059056723475676;
  shifts(10) =  0.456196152175878;

  // Read true solution from file
  TuckerOnNode::Tensor<scalar_t, memory_space> true_sol(dims);
  TuckerOnNode::import_tensor_binary(true_sol, "./gold_3x5x7x11_ss3.bin");

  // Call shift-scale
  int mode = 3;
  Tucker::transform_slices(tensor, mode, scales, shifts);

  // Checks
  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol, tol));
}