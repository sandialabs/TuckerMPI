/*
 * ttm_test_file.cpp
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

  // Read the tensor from a file
  Tucker::Tensor<scalar_t>* tensor =
      Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Read a matrix to multiply
  Tucker::Matrix<scalar_t>* mat =
      Tucker::importMatrix<scalar_t>("input_files/3x2.txt");

  // Read the true solution
  Tucker::Tensor<scalar_t>* trueSol =
      Tucker::importTensor<scalar_t>("input_files/3x2_mult_transp.txt");

  // Compute the TTM
  Tucker::Tensor<scalar_t>* mySol = Tucker::ttm(tensor, 0, mat, true);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);

  // Read a matrix to multiply
  mat = Tucker::importMatrix<scalar_t>("input_files/4x3.txt");

  // Read the true solution
  trueSol = Tucker::importTensor<scalar_t>("input_files/4x3_mult.txt");

  // Compute the TTM
  mySol = Tucker::ttm(tensor, 0, mat);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);

  // Read a matrix to multiply
  mat = Tucker::importMatrix<scalar_t>("input_files/5x8.txt");

  // Read the true solution
  trueSol = Tucker::importTensor<scalar_t>("input_files/5x8_mult_transp.txt");

  // Compute the TTM
  mySol = Tucker::ttm(tensor, 1, mat, true);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);

  // Read a matrix to multiply
  mat = Tucker::importMatrix<scalar_t>("input_files/2x5.txt");

  // Read the true solution
  trueSol = Tucker::importTensor<scalar_t>("input_files/2x5_mult.txt");

  // Compute the TTM
  mySol = Tucker::ttm(tensor, 1, mat);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);

  // Read a matrix to multiply
  mat = Tucker::importMatrix<scalar_t>("input_files/7x1.txt");

  // Read the true solution
  trueSol = Tucker::importTensor<scalar_t>("input_files/7x1_mult_transp.txt");

  // Compute the TTM
  mySol = Tucker::ttm(tensor, 2, mat, true);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);

  // Read a matrix to multiply
  mat = Tucker::importMatrix<scalar_t>("input_files/1x7.txt");

  // Read the true solution
  trueSol = Tucker::importTensor<scalar_t>("input_files/1x7_mult.txt");

  // Compute the TTM
  mySol = Tucker::ttm(tensor, 2, mat);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);

  // Read a matrix to multiply
  mat = Tucker::importMatrix<scalar_t>("input_files/11x25.txt");

  // Read the true solution
  trueSol = Tucker::importTensor<scalar_t>("input_files/11x25_mult_transp.txt");

  // Compute the TTM
  mySol = Tucker::ttm(tensor, 3, mat, true);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);

  // Read a matrix to multiply
  mat = Tucker::importMatrix<scalar_t>("input_files/17x11.txt");

  // Read the true solution
  trueSol = Tucker::importTensor<scalar_t>("input_files/17x11_mult.txt");

  // Compute the TTM
  mySol = Tucker::ttm(tensor, 3, mat);

  // Compare the computed solution to the true solution
  if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
  {
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(mat);
  Tucker::MemoryManager::safe_delete(mySol);
  Tucker::MemoryManager::safe_delete(trueSol);
  Tucker::MemoryManager::safe_delete(tensor);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


