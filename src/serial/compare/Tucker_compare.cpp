/*
 * Tucker_compare.cpp
 *
 *  Created on: Nov 28, 2016
 *      Author: amklinv
 */

#include<cassert>
#include<cstdlib>
#include "Tucker.hpp"

int main(int argc, char* argv[])
{
  // argv[1] is the number of entries
  // argv[2] is file 1
  // argv[3] is file 2
  // argv[4] is the tolerance
  // argv[5] is optional, should be 1(ignore sign is comparison) or 0(don't ignore), default to 0.
  assert(argc == 5 || argc == 6);

  // Get the number of entries and create the tensors
  // We store them as single dimensional arrays
  int nEntries = atoi(argv[1]);
  Tucker::SizeArray sz(1); sz[0] = nEntries;
  Tucker::Tensor<double> t1(sz);
  Tucker::Tensor<double> t2(sz);

  // Read the binary files
  Tucker::importTensorBinary(&t1, argv[2]);
  std::cout << "t1 norm2: " <<t1.norm2() << std::endl;
  Tucker::importTensorBinary(&t2, argv[3]);
  std::cout << "t2 norm2: " << t2.norm2() << std::endl;

  // for(int i=0; i< 100; i++){
  //   std::cout << t1.data()[i] << std::endl;
  // }
  // std::cout << std::endl;
  // Read the tolerance
  double tol = atof(argv[4]);
  bool approxEqual;
  if(argc == 6){
    if(atoi(argv[5]) == 1){
      approxEqual = isApproxEqual(&t1, &t2, tol, true, true);
    }
  }
  else{
    approxEqual = isApproxEqual(&t1, &t2, tol, true);
  } 

  if(approxEqual)
    return EXIT_SUCCESS;

  return EXIT_FAILURE;
}


