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

#include <vector>

#include "Tucker_Timer.hpp"
#include "Tucker_TuckerTensor.hpp"
#include "TuckerOnNode_ISVD.hpp"

namespace TuckerOnNode {

/** \brief A structure for storing a streaming %Tucker decomposition
 *
 * Allows users to access the data directly
 */
template<class scalar_t, class mem_space_t = Kokkos::DefaultExecutionSpace::memory_space>
class StreamingTuckerTensor {
public:
  using isvd_t = ISVD<scalar_t,mem_space_t>;
  using matrix_t = typename isvd_t::matrix_t;
  using tensor_t = Tensor<scalar_t, mem_space_t>;
  using ttensor_t = Tucker::TuckerTensor<tensor_t>;

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
   */
  StreamingTuckerTensor(const ttensor_t& X) :
    factorization(X), N(X.rank()), Gram(N), isvd(), squared_errors(N),
    Xnorm2(0.0) {}

  /** \brief Destructor
   */
  ~StreamingTuckerTensor() {}

  // The TuckerTensor factorization
  ttensor_t factorization;

  //! The number of dimensions
  int N;

  /** \brief Gram matrices
   */
  std::vector<matrix_t> Gram;

  /** \brief Pointer to ISVD factorization
   */
  isvd_t isvd;

  /** \brief Array of modal errors
   */
  std::vector<scalar_t> squared_errors;

  /** \brief Norm of full data tensor
   */
  scalar_t Xnorm2;

};

template <class scalar_t, class mem_space_t = Kokkos::DefaultExecutionSpace::memory_space>
StreamingTuckerTensor<scalar_t,mem_space_t>
StreamingSTHOSVD(const Tensor<scalar_t,mem_space_t>& X,
                 const Tucker::TuckerTensor< Tensor<scalar_t,mem_space_t> >& initial_factorization,
                 const TensorGramEigenvalues<scalar_t,mem_space_t>& initial_eigenvalues,
                 const char* filename,
                 const scalar_t epsilon,
                 Tucker::Timer &readTimer,
                 const std::string &streaming_stats_file,
                 bool useQR=false,
                 bool flipSign=false);

} // end namespace Tucker

#endif /* STREAMINGTUCKERTENSOR_HPP_ */
