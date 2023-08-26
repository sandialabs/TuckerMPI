/**
 * Originally import test here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/serial/tests/Tucker_shift_scale_test.cpp
 */

#include <gtest/gtest.h>
#include "TuckerOnNode.hpp"

template<class ScalarType, class ...Props1, class ...Props2>
bool checks(TuckerOnNode::Tensor<ScalarType, Props1...> t1,
            TuckerOnNode::Tensor<ScalarType, Props2...> t2,
            ScalarType tol)
{
  auto t1_h = Tucker::create_mirror_tensor_and_copy(Kokkos::HostSpace(), t1);
  auto t2_h = Tucker::create_mirror_tensor_and_copy(Kokkos::HostSpace(), t2);

  // 1) Owns any data
  if(t1_h.size() == 0 && t2_h.size() == 0){
    return true;
  }

  // 2) Same size
  if(t1_h.size() != t2_h.size()) {
    return false;
  }

  // 3) Values
  int numElements = t1_h.size();
  ScalarType errNorm2 = 0;
  for(int i=0; i<numElements; i++) {
    ScalarType err = std::abs(t1_h.data()[i] - t2_h.data()[i]);
    if(std::isnan(err)){
      return false;
    }
    errNorm2 += (err*err);
  }

  // 4) Tol
  ScalarType origNorm2 = t1_h.frobeniusNormSquared();
  ScalarType relErr = std::sqrt(errNorm2/origNorm2);
  if(relErr > tol){
    return false;
  }

  // Return
  return true;
}

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode0)
{
  using scalar_t = double;

  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  TuckerOnNode::read_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  // Define scales and shifts
  Kokkos::View<scalar_t*> scales("scales", 3);
  auto scales_h = Kokkos::create_mirror(scales);
  scales_h(0) = -0.258915944830978;
  scales_h(1) =  0.341369101972144;
  scales_h(2) =  0.357212764090606;
  Kokkos::deep_copy(scales, scales_h);

  Kokkos::View<scalar_t*> shifts("shifts", 3);
  auto shifts_h = Kokkos::create_mirror(shifts);
  shifts_h(0) = -0.450368098480610;
  shifts_h(1) = -0.408899824586946;
  shifts_h(2) =  0.094037031444121;
  Kokkos::deep_copy(shifts, shifts_h);

  // Read true solution from file
  TuckerOnNode::Tensor<scalar_t, Kokkos::HostSpace> true_sol_h(dims);
  TuckerOnNode::read_tensor_binary(true_sol_h, "./gold_3x5x7x11_ss0.bin");

  const int mode = 0;
  TuckerOnNode::transform_slices(tensor, mode, scales, shifts);

  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol_h, tol));
}

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode1)
{
  using scalar_t = double;

  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  TuckerOnNode::read_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  Kokkos::View<scalar_t*> scales("scales", 5);
  auto scales_h = Kokkos::create_mirror(scales);
  scales_h(0) =  0.262109709211147;
  scales_h(1) = -0.152432849551241;
  scales_h(2) = -0.038768240608501;
  scales_h(3) =  0.139323762199356;
  scales_h(4) =  0.417336040866845;
  Kokkos::deep_copy(scales, scales_h);

  Kokkos::View<scalar_t*> shifts("shifts", 5);
  auto shifts_h = Kokkos::create_mirror(shifts);
  shifts_h(0) =  0.463612200951355;
  shifts_h(1) = -0.011100213839396;
  shifts_h(2) = -0.279689899431367;
  shifts_h(3) = -0.273791359158714;
  shifts_h(4) =  0.036787804512826;
  Kokkos::deep_copy(shifts, shifts_h);

  TuckerOnNode::Tensor<scalar_t> true_sol_h(dims);
  TuckerOnNode::read_tensor_binary(true_sol_h, "./gold_3x5x7x11_ss1.bin");

  const int mode = 1;
  TuckerOnNode::transform_slices(tensor, mode, scales, shifts);

  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol_h, tol));
}

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode2)
{
  using scalar_t = double;
  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  TuckerOnNode::read_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  Kokkos::View<scalar_t*> scales("scales", 7);
  auto scales_h = Kokkos::create_mirror(scales);
  scales_h(0) =  0.133333580320122; scales_h(1) =  0.124000554312344;
  scales_h(2) = -0.172058403026751; scales_h(3) =  0.302965315958237;
  scales_h(4) =  0.499477858635892; scales_h(5) =  0.480978160932146;
  scales_h(6) = -0.372963057805143;
  Kokkos::deep_copy(scales, scales_h);

  Kokkos::View<scalar_t*> shifts("shifts", 7);
  auto shifts_h = Kokkos::create_mirror(shifts);
  shifts_h(0) = -0.338427426109669; shifts_h(1) =  0.215635404167474;
  shifts_h(2) =  0.077738876192409; shifts_h(3) = -0.066701022790881;
  shifts_h(4) =  0.384242782631094; shifts_h(5) = -0.106948244623087;
  shifts_h(6) = -0.321024847372268;
  Kokkos::deep_copy(shifts, shifts_h);

  TuckerOnNode::Tensor<scalar_t> true_sol_h(dims);
  TuckerOnNode::read_tensor_binary(true_sol_h, "./gold_3x5x7x11_ss2.bin");

  const int mode = 2;
  TuckerOnNode::transform_slices(tensor, mode, scales, shifts);

  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol_h, tol));
}

TEST(tuckerkokkos, transform_slices_3x5x7x11_mode3)
{
  using scalar_t = double;

  std::vector<int> dims = {3, 5, 7, 11};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);
  TuckerOnNode::read_tensor_binary(tensor, "./gold_3x5x7x11.bin");

  Kokkos::View<scalar_t*> scales("scales", 11);
  auto scales_h = Kokkos::create_mirror(scales);
  scales_h(0) = -0.375974083351957; scales_h(1) = -0.029236754133043;
  scales_h(2) =  0.356896327782193; scales_h(3) = -0.456609528433047;
  scales_h(4) =  0.191625145201306; scales_h(5) =  0.478985466675038;
  scales_h(6) = -0.216732101507863; scales_h(7) = -0.366219500005577;
  scales_h(8) =  0.185279684412687; scales_h(9) =  0.409454555749395;
  scales_h(10) = 0.110868982383243;
  Kokkos::deep_copy(scales, scales_h);

  Kokkos::View<scalar_t*> shifts("shifts", 11);
  auto shifts_h = Kokkos::create_mirror(shifts);
  shifts_h(0) = -0.267759854038207; shifts_h(1) = -0.476367533341775;
  shifts_h(2) =  0.107432610401855; shifts_h(3) = -0.389190678712850;
  shifts_h(4) = -0.092540492121601; shifts_h(5) =  0.384076806661962;
  shifts_h(6) =  0.048132777476588; shifts_h(7) = -0.130996923288383;
  shifts_h(8) = -0.291654017186653; shifts_h(9) = -0.059056723475676;
  shifts_h(10) =  0.456196152175878;
  Kokkos::deep_copy(shifts, shifts_h);

  TuckerOnNode::Tensor<scalar_t> true_sol_h(dims);
  TuckerOnNode::read_tensor_binary(true_sol_h, "./gold_3x5x7x11_ss3.bin");

  const int mode = 3;
  TuckerOnNode::transform_slices(tensor, mode, scales, shifts);

  scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();
  ASSERT_TRUE(checks(tensor, true_sol_h, tol));
}
