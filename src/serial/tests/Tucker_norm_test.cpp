/**
 * @file
 * \example
 *
 * \author Alicia Klinvex
 */

#include <cmath>
#include "Tucker.hpp"

int main()
{
  typedef double scalar_t; // specify precision

  const scalar_t TRUE_SOLUTION = 9.690249359274157;

  // Read the matrix from a binary file
  Tucker::Tensor<scalar_t>* tensor =
      Tucker::importTensor<scalar_t>("input_files/3x5x7x11.txt");

  // Compute its norm
  scalar_t norm = sqrt(tensor->norm2());

  // Free memory
  Tucker::MemoryManager::safe_delete(tensor);

  // Compare computed solution to true solution
  scalar_t diff = std::abs(norm-TRUE_SOLUTION);
  if(diff < 100 * std::numeric_limits<scalar_t>::epsilon())
    return EXIT_SUCCESS;
  else {
    std::cerr << "ERROR: The true solution is " << TRUE_SOLUTION
              << ", but the computed solution was " << norm
              << ", a difference of " << diff << std::endl;
  }

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    return EXIT_FAILURE;
  }

  return EXIT_FAILURE;
}

