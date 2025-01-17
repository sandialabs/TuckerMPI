/*
 * eig_test.cpp
 *
 *  Created on: Aug 30, 2016
 *      Author: amklinv
 */

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

  // Read the tensor from a binary file
  Tucker::Matrix<scalar_t>* matrix =
      Tucker::importMatrix<scalar_t>("input_files/5x5.txt");

  // Initialize the lower triangular portion of the matrix with NAN
  for(int r=0; r<5; r++) {
    for(int c=0; c<r; c++) {
      matrix->data()[r+5*c] = std::numeric_limits<scalar_t>::signaling_NaN();
    }
  }
  //Tucker::Matrix* matrix = Tucker::MemoryManager::safe_new<Tucker::Matrix>(30,30);
  // Compute the eigenpairs
  scalar_t* eigenvalues;
  //matrix->rand();
  //std::cout << matrix->prettyPrint() << std::endl;
  //Tucker::Matrix* S = Tucker::computeGram(matrix, 0);
  //std::cout << S->prettyPrint() << std::endl;
  Tucker::computeEigenpairs(matrix, eigenvalues, true);
  for(int i = 0; i < 30; i++){
    std::cout << eigenvalues[i] << std::endl;
  }

  scalar_t TRUE_EVALS[5] = {0.635794375487247, 0.355867612602428,
      0.061922384175464, -0.459221953743586, -0.690271293870763};

  scalar_t TRUE_EVECS[5*5] =
  {0.609179643854214, 0.461861259573850, 0.006538134339641, 0.226987202930453, 0.603339374584219,
   -0.042600532550886, 0.268064898175888, 0.673874553249279, 0.569936812817458, -0.383916033620496,
   0.752965729125975, -0.200041939033358, 0.180165297412201, -0.377310929868719, -0.467121784120092,
   -0.041273724014465, 0.734315518324370, -0.471551925483707, -0.110143371541715, -0.473903385229545,
   0.241711494571112, -0.368224563882684, -0.539471580774579, 0.684947982702565, -0.214016117322667
  };

  for(int i=0; i<5; i++) {
    scalar_t diff = std::abs(eigenvalues[i]-TRUE_EVALS[i]);
    if(diff > 100 * std::numeric_limits<scalar_t>::epsilon()) {
      std::cerr << "ERROR: The true eigenvalue is " << TRUE_EVALS[i]
                << ", but the computed eigenvalue was " << eigenvalues[i]
                << ", a difference of " << diff << std::endl;
      return EXIT_FAILURE;
    }
  }


  for(int i=0; i<5*5; i++) {
    scalar_t diff = std::abs(matrix->data()[i]-TRUE_EVECS[i]);
    if(diff > 100 * std::numeric_limits<scalar_t>::epsilon()) {
      std::cerr << "ERROR: The true solution is " << TRUE_EVECS[i]
                << ", but the computed solution was " << matrix->data()[i]
                << ", a difference of " << diff << std::endl;
      return EXIT_FAILURE;
    }
  }

  Tucker::MemoryManager::safe_delete(matrix);
  Tucker::MemoryManager::safe_delete_array(eigenvalues,5);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


