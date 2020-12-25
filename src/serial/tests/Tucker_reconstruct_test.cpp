/*
 * driver.cpp
 *
 *  Created on: Jun 3, 2016
 *      Author: Alicia Klinvex (amklinv@sandia.gov)
 */

#include "Tucker.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

template <class scalar_t>
bool approxEqual(scalar_t a, scalar_t b, scalar_t tol)
{
  if(std::abs(a-b) > tol)
    return false;
  return true;
}

int main()
{
  typedef double scalar_t; // specify precision

  Tucker::SizeArray* size =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(3);
  (*size)[0] = 2;
  (*size)[1] = 3;
  (*size)[2] = 5;
  Tucker::Tensor<scalar_t>* t =
      Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*size);
  scalar_t* data = t->data();
  for(int i=0; i<30; i++)
    data[i] = i+1;

  // set tolerance looser than 1e-4 (tightest possible for single precision)
  const struct Tucker::TuckerTensor<scalar_t>* factorization = Tucker::STHOSVD(t,(scalar_t) 1e-3);

  // Reconstruct the original tensor
  Tucker::Tensor<scalar_t>* temp = Tucker::ttm(factorization->G,0,
      factorization->U[0]);
  Tucker::Tensor<scalar_t>* temp2 = Tucker::ttm(temp,1,
        factorization->U[1]);
  Tucker::Tensor<scalar_t>* temp3 = Tucker::ttm(temp2,2,
        factorization->U[2]);

  scalar_t* newdata = temp3->data();
  // set elementwise tolerance looser than normwise tolerance
  for(int i=0; i<30; i++) {
    if(!approxEqual(newdata[i], data[i], (scalar_t) 1e-2)) {
      std::cerr << "data[" << i << "] should be " << data[i]
                << " but is " << newdata[i] << "; difference: "
                << data[i]-newdata[i] << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Free some memory
  Tucker::MemoryManager::safe_delete(t);
  Tucker::MemoryManager::safe_delete(size);
  Tucker::MemoryManager::safe_delete(factorization);
  Tucker::MemoryManager::safe_delete(temp);
  Tucker::MemoryManager::safe_delete(temp2);
  Tucker::MemoryManager::safe_delete(temp3);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
