/*
 * driver.cpp
 *
 *  Created on: Jun 3, 2016
 *      Author: Alicia Klinvex (amklinv@sandia.gov)
 */

#include "TuckerMPI.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[])
{
  bool approxEqual;

  // Initialize MPI
  MPI_Init(&argc,&argv);

  // Create the SizeArray
  Tucker::SizeArray size(3);
  size[0] = 2;
  size[1] = 3;
  size[2] = 5;

  // Create the processor grid
  Tucker::SizeArray nprocsPerDim(3);
  nprocsPerDim[0] = 1;
  nprocsPerDim[1] = 2;
  nprocsPerDim[2] = 2;

  // Create the distribution object
  TuckerMPI::Distribution dist(size,nprocsPerDim);

  // Create the tensor
  TuckerMPI::Tensor t(&dist);

  // Read the values from a file
  std::string filename = "input_files/ttm_data.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),&t);

  ///////////////////////////////////////////////////////////////////
  // First test: mode 0, no transpose
  ///////////////////////////////////////////////////////////////////

  // Create a matrix to multiply
  Tucker::Matrix mat(7,2);
  double* data = mat.data();
  data[0] = 131;  data[1] = 137; data[2] = 139;  data[3] = 149;
  data[4] = 151;  data[5] = 157; data[6] = 163;  data[7] = 167;
  data[8] = 173;  data[9] = 179; data[10] = 181; data[11] = 191;
  data[12] = 193; data[13] = 197;

  TuckerMPI::Tensor* result = TuckerMPI::ttm(&t,0,&mat,false);

  // Read the expected result from a file
  Tucker::SizeArray expectedSize(3);
  expectedSize[0] = 7;
  expectedSize[1] = 3;
  expectedSize[2] = 5;
  TuckerMPI::Distribution* expectedDist =
      Tucker::safe_new<TuckerMPI::Distribution>(expectedSize,nprocsPerDim);
  TuckerMPI::Tensor expectedResult(expectedDist);
  filename = "input_files/ttm_result0.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),&expectedResult);

  approxEqual = isApproxEqual(result,&expectedResult, 1e-10);
  delete result;
  if(!approxEqual) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  ///////////////////////////////////////////////////////////////////
  // Second test: mode 0, transpose
  ///////////////////////////////////////////////////////////////////

  // Create another matrix for the TTM
  Tucker::Matrix matt(2,7);
  data = matt.data();
  data[0] = 131; data[1] = 137; data[2] = 139; data[3] = 149;
  data[4] = 151; data[5] = 157; data[6] = 163; data[7] = 167;
  data[8] = 173; data[9] = 179; data[10] = 181; data[11] = 191;
  data[12] = 193; data[13] = 197;

  result = TuckerMPI::ttm(&t,0,&matt,true);

  // Read the expected result from a file
  expectedSize[0] = 7;
  expectedSize[1] = 3;
  expectedSize[2] = 5;
  expectedDist =
      Tucker::safe_new<TuckerMPI::Distribution>(expectedSize,nprocsPerDim);
  TuckerMPI::Tensor expectedResult2(expectedDist);
  filename = "input_files/ttm_result0t.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),&expectedResult2);

  approxEqual = isApproxEqual(result,&expectedResult2, 1e-10);
  delete result;
  if(!approxEqual) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  ///////////////////////////////////////////////////////////////////
  // Third test: mode 1, no transpose
  ///////////////////////////////////////////////////////////////////

  // Create another matrix for the TTM
  Tucker::Matrix mat1(7,3);
  data = mat1.data();
  data[0] = 131; data[1] = 137; data[2] = 139; data[3] = 149;
  data[4] = 151; data[5] = 157; data[6] = 163; data[7] = 167;
  data[8] = 173; data[9] = 179; data[10] = 181; data[11] = 191;
  data[12] = 193; data[13] = 197; data[14] = 199; data[15] = 211;
  data[16] = 223; data[17] = 227; data[18] = 229; data[19] = 233;
  data[20] = 239;

  result = TuckerMPI::ttm(&t,1,&mat1,false);

  // Read the expected result from a file
  expectedSize[0] = 2;
  expectedSize[1] = 7;
  expectedSize[2] = 5;
  expectedDist =
      Tucker::safe_new<TuckerMPI::Distribution>(expectedSize,nprocsPerDim);
  TuckerMPI::Tensor expectedResult3(expectedDist);
  filename = "input_files/ttm_result1.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),&expectedResult3);

  approxEqual = isApproxEqual(result,&expectedResult3, 1e-10);
  delete result;
  if(!approxEqual) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  ///////////////////////////////////////////////////////////////////
  // Fourth test: mode 1, transpose
  ///////////////////////////////////////////////////////////////////

  // Create another matrix for the TTM
  Tucker::Matrix mat1t(3,7);
  data = mat1t.data();
  data[0] = 131; data[1] = 137; data[2] = 139; data[3] = 149;
  data[4] = 151; data[5] = 157; data[6] = 163; data[7] = 167;
  data[8] = 173; data[9] = 179; data[10] = 181; data[11] = 191;
  data[12] = 193; data[13] = 197; data[14] = 199; data[15] = 211;
  data[16] = 223; data[17] = 227; data[18] = 229; data[19] = 233;
  data[20] = 239;

  result = TuckerMPI::ttm(&t,1,&mat1t,true);

  // Read the expected result from a file
  expectedSize[0] = 2;
  expectedSize[1] = 7;
  expectedSize[2] = 5;
  expectedDist =
      Tucker::safe_new<TuckerMPI::Distribution>(expectedSize,nprocsPerDim);
  TuckerMPI::Tensor expectedResult4(expectedDist);
  filename = "input_files/ttm_result1t.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),&expectedResult4);

  approxEqual = isApproxEqual(result,&expectedResult4, 1e-10);
  delete result;
  if(!approxEqual) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  ///////////////////////////////////////////////////////////////////
  // Fifth test: mode 2, no transpose
  ///////////////////////////////////////////////////////////////////

  // Create another matrix for the TTM
  Tucker::Matrix mat2(7,5);
  data = mat2.data();
  data[0] = 131; data[1] = 137; data[2] = 139; data[3] = 149;
  data[4] = 151; data[5] = 157; data[6] = 163; data[7] = 167;
  data[8] = 173; data[9] = 179; data[10] = 181; data[11] = 191;
  data[12] = 193; data[13] = 197; data[14] = 199; data[15] = 211;
  data[16] = 223; data[17] = 227; data[18] = 229; data[19] = 233;
  data[20] = 239; data[21] = 241; data[22] = 251; data[23] = 257;
  data[24] = 263; data[25] = 269; data[26] = 271; data[27] = 277;
  data[28] = 281; data[29] = 283; data[30] = 293; data[31] = 307;
  data[32] = 311; data[33] = 313; data[34] = 317;

  result = TuckerMPI::ttm(&t,2,&mat2,false);

  // Read the expected result from a file
  expectedSize[0] = 2;
  expectedSize[1] = 3;
  expectedSize[2] = 7;
  expectedDist =
      Tucker::safe_new<TuckerMPI::Distribution>(expectedSize,nprocsPerDim);
  TuckerMPI::Tensor expectedResult5(expectedDist);
  filename = "input_files/ttm_result2.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),&expectedResult5);

  approxEqual = isApproxEqual(result,&expectedResult5, 1e-10);
  delete result;
  if(!approxEqual) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  ///////////////////////////////////////////////////////////////////
  // Sixth test: mode 2, transpose
  ///////////////////////////////////////////////////////////////////

  // Create another matrix for the TTM
  Tucker::Matrix mat2t(5,7);
  data = mat2t.data();
  data[0] = 131; data[1] = 137; data[2] = 139; data[3] = 149;
  data[4] = 151; data[5] = 157; data[6] = 163; data[7] = 167;
  data[8] = 173; data[9] = 179; data[10] = 181; data[11] = 191;
  data[12] = 193; data[13] = 197; data[14] = 199; data[15] = 211;
  data[16] = 223; data[17] = 227; data[18] = 229; data[19] = 233;
  data[20] = 239; data[21] = 241; data[22] = 251; data[23] = 257;
  data[24] = 263; data[25] = 269; data[26] = 271; data[27] = 277;
  data[28] = 281; data[29] = 283; data[30] = 293; data[31] = 307;
  data[32] = 311; data[33] = 313; data[34] = 317;

  result = TuckerMPI::ttm(&t,2,&mat2t,true);

  // Read the expected result from a file
  expectedSize[0] = 2;
  expectedSize[1] = 3;
  expectedSize[2] = 7;
  expectedDist =
      Tucker::safe_new<TuckerMPI::Distribution>(expectedSize,nprocsPerDim);
  TuckerMPI::Tensor expectedResult6(expectedDist);
  filename = "input_files/ttm_result2t.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),&expectedResult6);

  approxEqual = isApproxEqual(result,&expectedResult6, 1e-10);
  delete result;
  if(!approxEqual) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

