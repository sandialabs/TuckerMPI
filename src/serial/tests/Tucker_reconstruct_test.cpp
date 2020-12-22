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

bool approxEqual(double a, double b, double tol)
{
  if(std::abs(a-b) > tol)
    return false;
  return true;
}

int main()
{
  Tucker::SizeArray* size =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(3);
  (*size)[0] = 2;
  (*size)[1] = 3;
  (*size)[2] = 5;
  Tucker::Tensor<double>* t =
      Tucker::MemoryManager::safe_new<Tucker::Tensor<double>>(*size);
  double* data = t->data();
  for(int i=0; i<30; i++)
    data[i] = i+1;

  const struct Tucker::TuckerTensor<double>* factorization = Tucker::STHOSVD(t,1e-6);

  // Reconstruct the original tensor
  Tucker::Tensor<double>* temp = Tucker::ttm(factorization->G,0,
      factorization->U[0]);
  Tucker::Tensor<double>* temp2 = Tucker::ttm(temp,1,
        factorization->U[1]);
  Tucker::Tensor<double>* temp3 = Tucker::ttm(temp2,2,
        factorization->U[2]);

  double* newdata = temp3->data();
  for(int i=0; i<30; i++) {
    if(!approxEqual(newdata[i], data[i], 1e-10)) {
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
