/** \copyright
 * Copyright (2016) Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 * certain rights in this software.
 * \n\n
 * BSD 2-Clause License
 * \n\n
 * Copyright (c) 2016, Sandia Corporation
 * All rights reserved.
 * \n\n
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * \n\n
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * \n\n
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * .
 * \n
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @file
 * \brief Contains functions relevant to computing a streaming %Tucker decomposition.
 *
 * \author Alicia Klinvex
 * \author Zitong Li
 * \author Saibal De
 * \author Hemanth Kolla
 */

#include "Tucker.hpp"
#include "Tucker_StreamingTuckerTensor.hpp"
#include <algorithm>
#include <limits>
#include <fstream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iomanip>

/** \namespace Tucker \brief Contains the data structures and functions
 * necessary for a sequential tucker decomposition
 */
namespace Tucker {

template <class scalar_t>
void updateStreamingGram(Matrix<scalar_t>* Gram, const Tensor<scalar_t>* Y, const int n)
{

  Matrix<scalar_t>* gram_to_add = computeGram(Y,n);
  size_t nnz = gram_to_add->getNumElements();
  for(int i=0; i<nnz; i++) {
    *(Gram->data()+i) += *(gram_to_add->data()+i);
  }
  MemoryManager::safe_delete<Matrix<scalar_t>>(gram_to_add);
}

template <class scalar_t>
const struct StreamingTuckerTensor<scalar_t>* StreamingHOSVD(const Tensor<scalar_t>* X, const TuckerTensor<scalar_t>* initial_factorization,
    const char* filename, const scalar_t epsilon, bool useQR, bool flipSign)
{

  // Create a struct to store the factorization
  struct StreamingTuckerTensor<scalar_t>* factorization = MemoryManager::safe_new<StreamingTuckerTensor<scalar_t>>(initial_factorization);

  int ndims = X->N();
  scalar_t tensorNorm = X->norm2();
  scalar_t thresh = epsilon*epsilon*tensorNorm/X->N();
  Tucker::SizeArray* slice_dims = MemoryManager::safe_new<SizeArray>((ndims-1));

  // Compute the initial Gram matrices for all non-streaming modes
  for(int n=0; n<ndims-1; n++) {
    factorization->Gram[n] = computeGram(X,n);
    (*slice_dims)[n] = X->size(n); 
  }

  //Create an object for the Tensor slice
  Tucker::Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*slice_dims); 

  //Open the file containing names of stream of snapshot files
  //Loop over, read each snapshot, update the Tucker model
  std::ifstream inStream(filename);
  std::string snapshot_file;

  while(inStream >> snapshot_file) {
    std::cout<< "Reading snaphot " << snapshot_file << std::endl;
    importTensorBinary(Y,snapshot_file.c_str());

    // Update Gram of non-streaming modes
    for(int n=0; n<ndims-1; n++) {
      updateStreamingGram(factorization->Gram[n], Y, n);
    }

    // Allocate memory for the new bases (factor matrices) along all modes
    Matrix<scalar_t>** U_new = MemoryManager::safe_new_array<Matrix<scalar_t>*>(ndims);

    // Update the bases (factor matrices) of non-streaming modes
    for(int n=0; n<ndims-1; n++) {
      computeEigenpairs(factorization->Gram[n], factorization->factorization->eigenvalues[n],
          U_new[n], thresh, flipSign);
    }

    // For the streaming mode initialize ISVD with full set of left singular vectors 
    //int numRows = X->size(ndims - 1);
    //U_new[ndims-1] = MemoryManager::safe_new<Matrix<scalar_t>>(numRows,numRows); 
    Matrix<scalar_t>* gram_last_mode = computeGram(X,ndims-1);
    computeEigenpairs(gram_last_mode, factorization->factorization->eigenvalues[ndims-1],
        U_new[ndims-1], 0.0 /* thresh */, flipSign);  
   
    ISVD<scalar_t>* iSVD = MemoryManager::safe_new<ISVD<scalar_t>>();
    //

    // TO DO

    //
    // Free memory
    //
    for (int n=0; n<ndims; n++) {
      MemoryManager::safe_delete(factorization->factorization->U[n]);
    }
    MemoryManager::safe_delete_array(factorization->factorization->U,ndims);

    // Set the factor matrices to the updated ones
    factorization->factorization->U = U_new;
    for (int n=0; n<ndims; n++) {
      factorization->factorization->U[n] = U_new[n];
    }

  }

  //Close the file containing snapshot filenames
  inStream.close();

  //
  //Free memory
  //
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(slice_dims);
  Tucker::MemoryManager::safe_delete<Tucker::Tensor<scalar_t>>(Y);

  return factorization;
}

template void updateStreamingGram(Matrix<float>*, const Tensor<float>*, const int);
template const struct StreamingTuckerTensor<float>* StreamingHOSVD(const Tensor<float>*, const TuckerTensor<float>*, 
             const char* filename, const float, bool, bool);

template void updateStreamingGram(Matrix<double>*, const Tensor<double>*, const int);
template const struct StreamingTuckerTensor<double>* StreamingHOSVD(const Tensor<double>*, const TuckerTensor<double>*,
             const char* filename, const double, bool, bool);

} // end namespace Tucker
