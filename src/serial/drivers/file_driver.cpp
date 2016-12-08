/*
 * driver.cpp
 *
 *  Created on: Jun 3, 2016
 *      Author: Alicia Klinvex (amklinv@sandia.gov)
 */

#include "Tucker.hpp"
#include <iostream>

int main()
{
  // Read tensor from file
  Tucker::Tensor* t = Tucker::importTensor("input_data/tensor11.txt");

  // Set the reduced size
  Tucker::SizeArray sz(5);
  sz[0] = 2;
  sz[1] = 3;
  sz[2] = 5;
  sz[3] = 11;
  sz[4] = 7;

  // Compute the factorization
  const struct Tucker::TuckerTensor* factorization = Tucker::STHOSVD(t,&sz,true);

  const Tucker::SizeArray& sizes = t->size();
  for(int i=0; i<sizes.size(); i++) {
    for(int j=0; j<sizes[i]; j++) {
      std::cout << "eigenvalues[" << i << "," << j << "] = " << factorization->eigenvalues[i][j] << std::endl;
    }
  }

  for(int i=0; i<sizes.size(); i++) {
    factorization->U[i]->print();
  }

  // Send the core to a file
  Tucker::exportTensor(factorization->G,"output_data/core11.txt");

  // Free some memory
  Tucker::safe_delete<const struct Tucker::TuckerTensor>(factorization);
  Tucker::safe_delete<Tucker::Tensor>(t);
}
