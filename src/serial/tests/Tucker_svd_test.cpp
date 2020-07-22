#include <cstdlib>
#include <cmath>
#include <limits>
#include "Tucker.hpp"
int main()
{
  Tucker::Matrix* matrix = Tucker::importMatrix("input_files/svd_test_input.txt");
  double trueSingularValues[5] = {2.726513883736618e+02, 87.476101096626150, 26.572191135796654,
   14.383954079111488, 6.338192024560426};
  Tucker::Matrix* U = Tucker::importMatrix("input_files/svd_test_outputU.txt");
  int nrows = matrix->nrows();
  int ncols = matrix->ncols();
  double* singularValues = Tucker::MemoryManager::safe_new_array<double>(nrows);
  Tucker::Matrix* singularVectors = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrows, ncols);
  Tucker::computeSVD(matrix, singularValues, singularVectors);

  for(int i=0; i<nrows; i++) {
    double diff = std::abs(singularValues[i]-trueSingularValues[i]);
    if(diff > 1e-10) {
      std::cerr << "ERROR: The true singular value is " << trueSingularValues[i]
                << ", but the computed singular value was " << singularValues[i]
                << ", a difference of " << diff << std::endl;
      return EXIT_FAILURE;
    }
  }

  for(int i=0; i<nrows*nrows; i++) {
    double diff = std::abs(std::abs(U->data()[i])-std::abs(singularVectors->data()[i]));
    if(diff > 1e-10) {
      std::cerr << "ERROR: The true solution is " << U->data()[i]
                << ", but the computed solution was " << singularVectors->data()[i]
                << ", a difference of " << diff << std::endl;
      return EXIT_FAILURE;
    }
  }
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(matrix);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(U);
  Tucker::MemoryManager::safe_delete_array<double>(singularValues, nrows);
  return EXIT_SUCCESS;
}