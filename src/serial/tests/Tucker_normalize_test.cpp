/*
 * normalize_test.cpp
 *
 *  Created on: Sep 1, 2016
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

  // Normalize the tensor
  Tucker::normalizeTensorMinMax(tensor,0);

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_mm0.txt");

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Normalize the tensor
  Tucker::normalizeTensorMinMax(tensor,1);

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_mm1.txt");

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Normalize the tensor
  Tucker::normalizeTensorMinMax(tensor,2);

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_mm2.txt");

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Normalize the tensor
  Tucker::normalizeTensorMinMax(tensor,3);

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_mm3.txt");

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Normalize the tensor
  Tucker::normalizeTensorStandardCentering(tensor,0,100 * std::numeric_limits<scalar_t>::epsilon());

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_sc0.txt");

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Normalize the tensor
  Tucker::normalizeTensorStandardCentering(tensor,1,100 * std::numeric_limits<scalar_t>::epsilon());

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_sc1.txt");

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Normalize the tensor
  Tucker::normalizeTensorStandardCentering(tensor,2,100 * std::numeric_limits<scalar_t>::epsilon());

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_sc2.txt");

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(tensor,true_sol,100 * std::numeric_limits<scalar_t>::epsilon())) {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(true_sol);

  // Read tensor from file
  tensor = Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Normalize the tensor
  Tucker::normalizeTensorStandardCentering(tensor,3,100 * std::numeric_limits<scalar_t>::epsilon());

  // Read true solution from file
  true_sol = Tucker::importTensor<scalar_t>("input_files/3x5x7x11_sc3.txt");

  // Compare the computed solution to the true solution
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
