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
#include <iostream>

/** \namespace Tucker \brief Contains the data structures and functions
 * necessary for a sequential tucker decomposition
 */
namespace Tucker {

template <class scalar_t>
const struct StreamingTuckerTensor<scalar_t>* StreamingSTHOSVD(const Tensor<scalar_t>* X,
                                                               const TuckerTensor<scalar_t>* initial_factorization,
                                                               const char* filename,
                                                               const scalar_t epsilon,
                                                               Timer &readTimer,
                                                               bool useQR,
                                                               bool flipSign)
{

  // Create a struct to store the factorization
  struct StreamingTuckerTensor<scalar_t>* factorization = MemoryManager::safe_new<StreamingTuckerTensor<scalar_t>>(initial_factorization);

  // Construct and initialize ISVD object
  factorization->isvd->initializeFactors(factorization->factorization);

  // Track norm of data tensor
  int ndims = X->N();
  factorization->Xnorm2 = X->norm2();

  // Compute errors
  for (int n = 0; n < ndims - 1; ++n) {
    factorization->squared_errors[n] = 0;
    for (int i = factorization->factorization->G->size(n); i < factorization->factorization->U[n]->nrows(); ++i) {
      if (factorization->factorization->eigenvalues && factorization->factorization->eigenvalues[n]) {
        factorization->squared_errors[n] += std::abs(factorization->factorization->eigenvalues[n][i]);
      } else if (factorization->factorization->singularValues && factorization->factorization->singularValues[n]) {
        factorization->squared_errors[n] += std::pow(factorization->factorization->singularValues[n][i], 2);
      } else {
        // TODO - encountered null eigenvalues and null singular values - throw exception?
      }
    }
  }
  factorization->squared_errors[ndims - 1] = std::pow(factorization->isvd->getErrorNorm(), 2);

  // open status file
  std::ofstream log_stream("stats_stream.txt");
  log_stream << std::scientific;

  // print status header
  {
    for (int n = 0; n < ndims; ++n) {
      std::ostringstream str;
      str << "N[" << n << "]";
      log_stream << std::setw(6) << str.str() << " " << std::flush;
    }
    log_stream << std::setw(12) << "datanorm" << " " << std::flush;
    for (int n = 0; n < ndims; ++n) {
      std::ostringstream str;
      str << "R[" << n << "]";
      log_stream << std::setw(6) << str.str() << " " << std::flush;
    }
    for (int n = 0; n < ndims; ++n) {
      std::ostringstream str;
      str << "abserr[" << n << "]";
      log_stream << std::setw(12) << str.str() << " " << std::flush;
    }
    log_stream << std::setw(12) << "abserr" << " " << std::flush;
    log_stream << std::setw(12) << "relerr" << std::endl;
  }

  // print status after initial ST-HOSVD
  {
    for (int n = 0; n < ndims; ++n) {
      log_stream << std::setw(6) << factorization->factorization->U[n]->nrows() << " " << std::flush;
    }
    log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization->Xnorm2) << " " << std::flush;
    for (int n = 0; n < ndims; ++n) {
      log_stream << std::setw(6) << factorization->factorization->U[n]->ncols() << " " << std::flush;
    }
    scalar_t Enorm2 = 0;
    for (int n = 0; n < ndims; ++n) {
      Enorm2 += factorization->squared_errors[n];
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization->squared_errors[n]) << " " << std::flush;
    }
    log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2) << " " << std::flush;
    log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2 / factorization->Xnorm2) << std::endl;
  }

  // Compute the slice dimensions corresponding to non-streaming modes
  Tucker::SizeArray* slice_dims = MemoryManager::safe_new<SizeArray>((ndims-1));
  for(int n=0; n<ndims-1; n++) {
    (*slice_dims)[n] = X->size(n); 
  }

  //Open the file containing names of stream of snapshot files
  //Loop over, read each snapshot, update the Tucker model
  std::ifstream inStream(filename);
  std::string snapshot_file;
  
  while(inStream >> snapshot_file) {
    // Read the new tensor slice
    std::cout<< "Reading snapshot " << snapshot_file << std::endl;
    readTimer.start();
    Tucker::Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*slice_dims);
    importTensorBinary(Y,snapshot_file.c_str());
    readTimer.stop();

    // compute/update data norms
    const scalar_t Ynorm2 = Y->norm2();
    factorization->Xnorm2 += Ynorm2;

    // compute total allowed error based off current error
    scalar_t tolerance = 0;
    {
      for (int n = 0; n < ndims; ++n) {
        tolerance += factorization->squared_errors[n];
      }

      tolerance = epsilon * epsilon * factorization->Xnorm2 - tolerance;
    }

    // compute allowed error for non-streaming modes
    const scalar_t thresh = tolerance / ndims;

    // Loop over non-streaming modes
    for(int n=0; n<ndims-1; n++) {
      // Line 2 of streaming STHOSVD update
      // compute projection of new slice onto existing basis
      Tensor<scalar_t> *D;
      {
        // D = Y x_n U[n].T
        const bool isTransposed = true;
        D = ttm(Y, n, factorization->factorization->U[n], isTransposed);
      }

      // Line 3 of streaming STHOSVD update
      // compute orthogonal complement of new slice w.r.t. existing basis
      Tensor<scalar_t> *E;
      {
        // E = D x_n U[n]
        const bool isTransposed = false;
        E = ttm(D, n, factorization->factorization->U[n], isTransposed);
      }
      {
        // E = -E
        const int nelm = E->getNumElements();
        const scalar_t alpha = -1;
        const int incr = 1;
        scal(&nelm, &alpha, E->data(), &incr);
      }
      {
        // E = Y + E
        const int nelm = E->getNumElements();
        const scalar_t alpha = 1;
        const int incr = 1;
        axpy(&nelm, &alpha, Y->data(), &incr, E->data(), &incr);
      }

      const scalar_t Enorm2 = E->norm2();

      if (Enorm2 <= thresh) {
        factorization->squared_errors[n] += Enorm2;
        tolerance -= Enorm2;
        MemoryManager::safe_delete(E);

        std::swap(Y, D);
        MemoryManager::safe_delete(D);
      } else {
        // Line 4 of streaming STHOSVD update
        // compute Gram of orthogonal complement of new slice
        Matrix<scalar_t> *G = computeGram(E, n);

        // Lines 5-8 of streaming STHOSVD update
        // compute truncated eigendecomposition of Gram
        scalar_t *eigenvalues;
        Matrix<scalar_t> *V;
        computeEigenpairs(G, eigenvalues, V, thresh);

        const int R_new = V->ncols();
        for (int i = R_new; i < V->nrows(); ++i) {
          factorization->squared_errors[n] += std::abs(eigenvalues[i]);
          tolerance -= std::abs(eigenvalues[i]);
        }

        MemoryManager::safe_delete(G);
        MemoryManager::safe_delete_array(eigenvalues, V->nrows());

        // Line 9 of streaming STHOSVD update
        // project orthgonal complement to new basis
        Tensor<scalar_t> *K;
        {
          // K = E x_n V.T
          const bool isTransposed = true;
          K = ttm(E, n, V, isTransposed);
        }
        MemoryManager::safe_delete(E);

        // Line 10 of streaming STHOSVD update
        // pad core with zeros
        {
          Tensor<scalar_t> *core = padTensorAlongMode(factorization->factorization->G, n, R_new);
          std::swap(factorization->factorization->G, core);
          MemoryManager::safe_delete(core);
        }

        // Line 11 of streaming STHOSVD update
        // pad ISVD right singular vectors with zeros
        factorization->isvd->padRightSingularVectorsAlongMode(n, R_new);

        // Line 12 of streaming STHOSVD algorithm
        // prepare slice for next mode
        Tensor<scalar_t> *Y_new = concatenateTensorsAlongMode(D, K, n);
        std::swap(Y, Y_new);
        MemoryManager::safe_delete(D);
        MemoryManager::safe_delete(K);
        MemoryManager::safe_delete(Y_new);

        // Line 13 of streaming STHOSVD algorithm
        // update basis
        Matrix<scalar_t> *U_new;
        {
          const Matrix<scalar_t> *U = factorization->factorization->U[n];
          const int N = U->nrows();
          const int R = U->ncols();
          U_new = MemoryManager::safe_new<Matrix<scalar_t>>(N, R + R_new);
          {
            const int nelm = N * R;
            const int incr = 1;
            copy(&nelm, U->data(), &incr, U_new->data(), &incr);
          }
          {
            const int nelm = N * R_new;
            const int incr = 1;
            copy(&nelm, V->data(), &incr, U_new->data() + N * R, &incr);
          }
        }
        MemoryManager::safe_delete(V);

        std::swap(factorization->factorization->U[n], U_new);
        MemoryManager::safe_delete(U_new);
      }
    }

    // Line 16 of streaming STHOSVD update algorithm
    // Add new row to ISVD factorization
    const scalar_t delta = std::sqrt(tolerance / Y->norm2());
    factorization->isvd->updateFactorsWithNewSlice(Y, delta);

    factorization->squared_errors[ndims - 1] = std::pow(factorization->isvd->getErrorNorm(), 2);

    // Lines 17 of streaming STHOSVD update algorithm
    // Retrieve updated left singular vectors from ISVD factorization
    Matrix<scalar_t> *U_new = nullptr;
    {
      const Matrix<scalar_t> *U_isvd = factorization->isvd->getLeftSingularVectors();
      const int &nrows = U_isvd->nrows();
      const int &ncols = U_isvd->ncols();
      const int &nelem = nrows * ncols;
      const int &ONE = 1;
      U_new = MemoryManager::safe_new<Matrix<scalar_t>>(nrows, ncols);
      copy(&nelem, U_isvd->data(), &ONE, U_new->data(), &ONE);
    }

    // Line 22 of StreamingTuckerUpdate algorithm
    // Split U_new into two submatrices
    // Use first submatrix to update core
    {
      const int m_new = U_new->nrows();
      const int r_new = U_new->ncols();

      const int m_old = m_new - 1;
      const int r_old = factorization->factorization->U[ndims - 1]->ncols();

      Matrix<scalar_t> *M = MemoryManager::safe_new<Matrix<scalar_t>>(r_new, r_old);
      {
        const char transa = 'T';
        const char transb = 'N';
        const scalar_t alpha = 1;
        const scalar_t beta = 0;
        gemm(&transa, &transb,
             &r_new, &r_old, &m_old,
             &alpha,
             U_new->data(), &m_new,
             factorization->factorization->U[ndims - 1]->data(), &m_old,
             &beta,
             M->data(), &r_new);
      }

      Tensor<scalar_t> *temp3 = ttm(factorization->factorization->G, ndims - 1, M);
      std::swap(factorization->factorization->G, temp3);

      MemoryManager::safe_delete(M);
      MemoryManager::safe_delete(temp3);
    }

    // Lines 23-24 of StreamingTuckerUpdate algorithm
    // Use last row of Unew[d] to update scale new slice and update core in-place
    {
      const int &nrow = Y->getNumElements();
      const int &ncol = U_new->nrows();
      const int &rank = U_new->ncols();
      const int &ONE = 1;
      for (int j = 0; j < rank; ++j) {
        axpy(&nrow,
             U_new->data() + (j + 1) * ncol - 1,
             Y->data(), &ONE,
             factorization->factorization->G->data() + j * nrow, &ONE);
      }
    }

    // Line 25 of StreamingTuckerUpdate algorithm
    // Save new factor matrix
    MemoryManager::safe_delete(factorization->factorization->U[ndims - 1]);
    factorization->factorization->U[ndims - 1] = U_new;

    // Free memory
    MemoryManager::safe_delete<Tucker::Tensor<scalar_t>>(Y);

    // print status after streaming ST-HOSVD update
    {
      for (int n = 0; n < ndims; ++n) {
        log_stream << std::setw(6) << factorization->factorization->U[n]->nrows() << " " << std::flush;
      }
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization->Xnorm2) << " " << std::flush;
      for (int n = 0; n < ndims; ++n) {
        log_stream << std::setw(6) << factorization->factorization->U[n]->ncols() << " " << std::flush;
      }
      scalar_t Enorm2 = 0;
      for (int n = 0; n < ndims; ++n) {
        Enorm2 += factorization->squared_errors[n];
        log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization->squared_errors[n]) << " " << std::flush;
      }
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2) << " " << std::flush;
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2 / factorization->Xnorm2) << std::endl;
    }
  }

  // Close the file containing snapshot filenames
  inStream.close();

  // Free memory
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(slice_dims);

  return factorization;
}

// Explicit instantiations

template class StreamingTuckerTensor<float>;
template class StreamingTuckerTensor<double>;

template const struct StreamingTuckerTensor<float>* StreamingSTHOSVD(const Tensor<float>*,
                                                                     const TuckerTensor<float>*, 
                                                                     const char* filename,
                                                                     const float,
                                                                     Timer &,
                                                                     bool,
                                                                     bool);

template const struct StreamingTuckerTensor<double>* StreamingSTHOSVD(const Tensor<double>*,
                                                                      const TuckerTensor<double>*,
                                                                      const char* filename,
                                                                      const double,
                                                                      Timer &,
                                                                      bool,
                                                                      bool);

} // end namespace Tucker
