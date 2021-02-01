#include <cstdlib>
#include <cmath>
#include <limits>
#include "Tucker.hpp"
int main()
{

// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t; 
#else
  typedef double scalar_t;
#endif

  Tucker::Matrix<scalar_t>* matrix = Tucker::importMatrix<scalar_t>("input_files/svd_test_input.txt");
  scalar_t trueSingularValues[5] = {2.726513883736618e+02, 87.476101096626150, 26.572191135796654,
   14.383954079111488, 6.338192024560426};
  Tucker::Matrix<scalar_t>* U = Tucker::importMatrix<scalar_t>("input_files/svd_test_outputU.txt");
  int nrows = matrix->nrows();
  int ncols = matrix->ncols();
  scalar_t* singularValues = Tucker::MemoryManager::safe_new_array<scalar_t>(nrows);
  Tucker::Matrix<scalar_t>* singularVectors = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(nrows, ncols);
  Tucker::computeSVD(matrix, singularValues, singularVectors);

  for(int i=0; i<nrows; i++) {
    scalar_t diff = std::abs(singularValues[i]-trueSingularValues[i]);
    if(diff > 1000 * std::numeric_limits<scalar_t>::epsilon()) {
      std::cerr << "ERROR: The true singular value is " << trueSingularValues[i]
                << ", but the computed singular value was " << singularValues[i]
                << ", a difference of " << diff << std::endl;
      return EXIT_FAILURE;
    }
  }

  for(int i=0; i<nrows*nrows; i++) {
    scalar_t diff = std::abs(std::abs(U->data()[i])-std::abs(singularVectors->data()[i]));
    if(diff > 1000 * std::numeric_limits<scalar_t>::epsilon()) {
      std::cerr << "ERROR: The true solution is " << U->data()[i]
                << ", but the computed solution was " << singularVectors->data()[i]
                << ", a difference of " << diff << std::endl;
      return EXIT_FAILURE;
    }
  }
  Tucker::MemoryManager::safe_delete(matrix);
  Tucker::MemoryManager::safe_delete(U);
  Tucker::MemoryManager::safe_delete_array(singularValues, nrows);
  return EXIT_SUCCESS;
}