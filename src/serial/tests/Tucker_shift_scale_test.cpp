/*
 * shift_scale_test.cpp
 *
 *  Created on: Aug 31, 2016
 *      Author: amklinv
 */

#include "Tucker.hpp"

int main()
{

// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t; 
#else
  typedef double scalar_t;
#endif

  Tucker::Tensor<scalar_t>* tensor;
  Tucker::Tensor<scalar_t>* true_sol;

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  scalar_t shifts0[3] = {-0.450368098480610,
      -0.408899824586946, 0.094037031444121};

  scalar_t scales0[3] = {-0.258915944830978,
      0.341369101972144, 0.357212764090606};

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_ss0.txt");

  // Call shift-scale
  Tucker::transformSlices(tensor,0,scales0,shifts0);

  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  scalar_t shifts1[5] = {0.463612200951355, -0.011100213839396,
      -0.279689899431367, -0.273791359158714, 0.036787804512826};

  scalar_t scales1[5] = {0.262109709211147, -0.152432849551241,
      -0.038768240608501, 0.139323762199356, 0.417336040866845};

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_ss1.txt");

  // Call shift-scale
  Tucker::transformSlices(tensor,1,scales1,shifts1);

  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  scalar_t shifts2[7] = {-0.338427426109669, 0.215635404167474,
      0.077738876192409, -0.066701022790881, 0.384242782631094,
     -0.106948244623087, -0.321024847372268};

  scalar_t scales2[7] = {0.133333580320122, 0.124000554312344,
     -0.172058403026751, 0.302965315958237, 0.499477858635892,
      0.480978160932146, -0.372963057805143};

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_ss2.txt");

  // Call shift-scale
  Tucker::transformSlices(tensor,2,scales2,shifts2);

  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  scalar_t shifts3[11] = {-0.267759854038207, -0.476367533341775,
       0.107432610401855, -0.389190678712850, -0.092540492121601,
       0.384076806661962, 0.048132777476588, -0.130996923288383,
      -0.291654017186653, -0.059056723475676, 0.456196152175878};

  scalar_t scales3[11] = {-0.375974083351957, -0.029236754133043,
       0.356896327782193, -0.456609528433047, 0.191625145201306,
       0.478985466675038, -0.216732101507863, -0.366219500005577,
       0.185279684412687, 0.409454555749395, 0.110868982383243};

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_ss3.txt");

  // Call shift-scale
  Tucker::transformSlices(tensor,3,scales3,shifts3);

  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


