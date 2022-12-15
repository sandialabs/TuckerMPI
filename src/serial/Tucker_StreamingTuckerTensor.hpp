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
 * \brief Classes and functions for %Tucker decomposition on data streaming along the last mode
 *
 *
 * \author Alicia Klinvex
 * \author Zitong Li
 * \author Saibal De
 * \author Hemanth Kolla
 */

#ifndef STREAMINGTUCKERTENSOR_HPP_
#define STREAMINGTUCKERTENSOR_HPP_

#include "Tucker_TuckerTensor.hpp"
#include "Tucker_ISVD.hpp"

namespace Tucker {

/** \brief A structure for storing a streaming %Tucker decomposition
 *
 * Allows users to access the data directly
 */
template<class scalar_t>
class StreamingTuckerTensor {
public:
  /** \brief Constructor
   *
   * Stores a copy of a TuckerTensor object, and allocates memory for 
   * gram matrices for all but the last (streaming) mode.
   * \param[in] X Pointer to TuckerTensor object
   *
   * \note The TuckerTensor passed in at the time of construction 
   * is the initial factorization. This is updated based on streaming slices.
   * Since the TuckerTensor class does not allocate memory for the Gram matrices
   * that is done by this class
   *
   * \exception std::runtime_error \a !X 
   */
  StreamingTuckerTensor(const TuckerTensor<scalar_t>* X)
  {
    if(!X) {
      std::ostringstream oss;
      oss << "Tucker::StreamingTuckerTensor::StreamingTuckerTensor(const TuckerTensor<scalar_t>* X): X not allocated";
      throw std::runtime_error(oss.str());
    }

    // const_cast below is due to a legacy issue. 
    // STHOSVD which is originally used to compute the Tucker factorization 
    // returns const Tucker_Tensor<scalar_t> * type. But the underlying
    // data members are non const pointers (core tensor, factor matrices, etc).
    // So making a copy of those underlying (non const pointer) data types will
    // potentially have dangling pointers. Best to just copy the
    // Tucker_Tensor<scalar_t> pointer itself, and delete it in the destructor.  
    factorization = const_cast<TuckerTensor<scalar_t>*>(X);
    N = X->N;
    Gram = MemoryManager::safe_new_array<Matrix<scalar_t>*>(N-1);
    isvd = MemoryManager::safe_new<ISVD<scalar_t>>();
    squared_errors = MemoryManager::safe_new_array<scalar_t>(N);
    Xnorm2 = 0;
  }

  /** \brief Destructor
   *
   * Deallocates memory for the core tensor, the factor matrices,
   * the eigenvalues, and the timers.
   */
  ~StreamingTuckerTensor()
  {
    if(factorization) MemoryManager::safe_delete(factorization);
    for(int i=0; i<N-1; i++) {
      if(Gram[i]) MemoryManager::safe_delete(Gram[i]);
    }
    MemoryManager::safe_delete_array(Gram,N-1);
    MemoryManager::safe_delete(isvd);
    MemoryManager::safe_delete_array(squared_errors, N);
  }

  // The TuckerTensor factorization
  TuckerTensor<scalar_t>* factorization;


  /** \brief Gram matrices
   *
   * #Gram[n] is a pointer to the Gram matrix of n-th mode
   *
   * \note These are stored as pointers because Matrix has no
   * default constructor.
   */
  Matrix<scalar_t>** Gram;

  /** \brief Pointer to ISVD factorization
   */
  ISVD<scalar_t> *isvd;

  /** \brief Norm of full data tensor
   */
  scalar_t Xnorm2;

  /** \brief Array of modal errors
   */
  scalar_t *squared_errors;

  //! The number of dimensions
  int N;

private:
  /// @cond EXCLUDE
  StreamingTuckerTensor(StreamingTuckerTensor<scalar_t>& stt);
  /// @endcond

};

template <class scalar_t>
const struct StreamingTuckerTensor<scalar_t>* StreamingHOSVD(const Tensor<scalar_t>* X, const TuckerTensor<scalar_t>* initial_factorization,
    const char* filename, const scalar_t epsilon, bool useQR=false, bool flipSign=false);

} // end namespace Tucker

#endif /* STREAMINGTUCKERTENSOR_HPP_ */
