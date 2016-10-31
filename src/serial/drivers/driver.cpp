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
  Tucker::SizeArray size(4);
  for(int i=0; i<4; i++)
    size[i] = 2;
  Tucker::Tensor t(size);
  double* data = t.data();
  for(int i=0; i<16; i++)
    data[i] = i+1;

  const struct Tucker::TuckerTensor* factorization = Tucker::STHOSVD(&t,1e-6);

  // Output the tensor
  std::cout << "Tensor:\n";
  factorization->G->print();

  // Output the factors
  for(int i=0; i<factorization->N; i++) {
    std::cout << "Factor " << i << ":\n";
    factorization->U[i]->print();
  }

  // Free some memory
  delete factorization;
}

/*
 * Output should be:
 *
 * Tensor:
 * data[0] = -0.041808
 * data[1] = 0.0631846
 * data[2] = 0.146181
 * data[3] = -0.167471
 * data[4] = 0.302263
 * data[5] = -0.344384
 * data[6] = -0.795369
 * data[7] = 0.0347241
 * data[8] = 0.609435
 * data[9] = -0.69318
 * data[10] = -1.60091
 * data[11] = 0.0576784
 * data[12] = -3.2914
 * data[13] = 0.0351388
 * data[14] = 0.0205104
 * data[15] = 38.4818
 * Factor 0:
 * elements[0] = -0.738586
 * elements[1] = 0.674159
 * elements[2] = 0.674159
 * elements[3] = 0.738586
 * Factor 1:
 * elements[0] = -0.768982
 * elements[1] = 0.639271
 * elements[2] = 0.639271
 * elements[3] = 0.768982
 * Factor 2:
 * elements[0] = -0.827501
 * elements[1] = 0.561464
 * elements[2] = 0.561464
 * elements[3] = 0.827501
 * Factor 3:
 * elements[0] = -0.933184
 * elements[1] = 0.359399
 * elements[2] = 0.359399
 * elements[3] = 0.933184
 */

