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

#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas3_gemm.hpp>

#include "TuckerOnNode.hpp"
#include "TuckerOnNode_StreamingTuckerTensor.hpp"
#include "TuckerOnNode_compute_gram.hpp"
#include "Tucker_create_mirror.hpp"

/** \namespace Tucker \brief Contains the data structures and functions
 * necessary for a sequential tucker decomposition
 */
namespace TuckerOnNode {

template <class scalar_t, class mem_space_t>
StreamingTuckerTensor<scalar_t,mem_space_t>
StreamingSTHOSVD(
  const Tensor<scalar_t,mem_space_t>& X,
  const Tucker::TuckerTensor< Tensor<scalar_t,mem_space_t> >& initial_factorization,
  const TensorGramEigenvalues<scalar_t,mem_space_t>& initial_eigenvalues,
  const char* filename,
  const scalar_t epsilon,
  const std::string &streaming_stats_file,
  bool useQR,
  bool flipSign)
{
  using isvd_t = ISVD<scalar_t,mem_space_t>;
  using matrix_t = typename isvd_t::matrix_t;
  using tensor_t = Tensor<scalar_t, mem_space_t>;
  using ttensor_t = Tucker::TuckerTensor<tensor_t>;
  using exec_space = typename mem_space_t::execution_space;

  // Create a struct to store the factorization
  StreamingTuckerTensor<scalar_t,mem_space_t> factorization(initial_factorization);

  // Construct and initialize ISVD object
  factorization.isvd.initializeFactors(factorization.factorization,
                                       initial_eigenvalues);

  // Core tensor
  auto G = factorization.factorization.coreTensor();

  // Create array of factor matrices.  This is required because TuckerTensor
  // stores them in one contiguous array, so adding new columns is extremly
  // difficult
  int ndims = X.rank();
  std::vector<matrix_t> U(ndims);
  for (int i=0; i<ndims; ++i)
    U[i] = factorization.factorization.factorMatrix(i);

  // Track norm of data tensor
  factorization.Xnorm2 = X.frobeniusNormSquared();

  // Compute errors
  for (int n = 0; n < ndims - 1; ++n) {
    factorization.squared_errors[n] = 0;
    auto eigs = initial_eigenvalues[n];
    for (int i = G.extent(n); i < U[n].extent(0); ++i) {
      if (eigs[i]) {
        factorization.squared_errors[n] += std::abs(eigs[i]);
      } else {
        // TODO - encountered null eigenvalues and null singular values - throw exception?
      }
    }
  }
  factorization.squared_errors[ndims - 1] = std::pow(factorization.isvd.getErrorNorm(), 2);

  // open status file
  std::ofstream log_stream(streaming_stats_file);
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
      log_stream << std::setw(6) << U[n].extent(0) << " " << std::flush;
    }
    log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization.Xnorm2) << " " << std::flush;
    for (int n = 0; n < ndims; ++n) {
      log_stream << std::setw(6) << U[n].extent(1) << " " << std::flush;
    }
    scalar_t Enorm2 = 0;
    for (int n = 0; n < ndims; ++n) {
      Enorm2 += factorization.squared_errors[n];
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization.squared_errors[n]) << " " << std::flush;
    }
    log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2) << " " << std::flush;
    log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2 / factorization.Xnorm2) << std::endl;
  }

  // Compute the slice dimensions corresponding to non-streaming modes
  auto slice_dims = Kokkos::subview(X.dimensionsOnHost(), std::make_pair(0,ndims-1));

  //Open the file containing names of stream of snapshot files
  //Loop over, read each snapshot, update the Tucker model
  std::ifstream inStream(filename);
  std::string snapshot_file;

  while(inStream >> snapshot_file) {
    // Read the new tensor slice
    std::cout<< "Reading snapshot " << snapshot_file << std::endl;
    Tensor<scalar_t, Kokkos::HostSpace> Yh(slice_dims);
    read_tensor_binary(Yh, snapshot_file);
    tensor_t Y = Tucker::create_mirror_tensor_and_copy(mem_space_t(), Yh);

    // compute/update data norms
    const scalar_t Ynorm2 = Y.frobeniusNormSquared();
    factorization.Xnorm2 += Ynorm2;

    // compute total allowed error based off current error
    scalar_t tolerance = 0;
    {
      for (int n = 0; n < ndims; ++n) {
        tolerance += factorization.squared_errors[n];
      }

      tolerance = epsilon * epsilon * factorization.Xnorm2 - tolerance;
    }

    // compute allowed error for non-streaming modes
    const scalar_t thresh = tolerance / ndims;

    // Loop over non-streaming modes
    for(int n=0; n<ndims-1; n++) {

      // Line 2 of streaming STHOSVD update
      // compute projection of new slice onto existing basis
      // D = Y x_n U[n].T
      tensor_t D = ttm(Y, n, U[n], true);

      // Line 3 of streaming STHOSVD update
      // compute orthogonal complement of new slice w.r.t. existing basis
      // E = D x_n U[n]
      tensor_t E = ttm(D, n, U[n], false);

      // E = Y-E
      KokkosBlas::axpby(scalar_t(1.0), Y.data(), scalar_t(-1.0), E.data());

      const scalar_t Enorm2 = E.frobeniusNormSquared();

      if (Enorm2 <= thresh) {
        factorization.squared_errors[n] += Enorm2;
        tolerance -= Enorm2;

        Y = D;
      } else {
        // Line 4 of streaming STHOSVD update
        // compute Gram of orthogonal complement of new slice
        matrix_t gram = compute_gram(E, n);

        // Lines 5-8 of streaming STHOSVD update
        // compute truncated eigendecomposition of Gram
        auto eigenvalues = Tucker::impl::compute_and_sort_descending_eigvals_and_eigvecs_inplace(gram, false); // on host
        int nev = Tucker::impl::count_eigvals_using_threshold(eigenvalues, thresh);
        matrix_t V = Kokkos::subview(gram, Kokkos::ALL, std::make_pair(0, nev));

        const int R_new = V.extent(1);
        for (int i=R_new; i<V.extent(0); ++i) {
          factorization.squared_errors[n] += std::abs(eigenvalues[i]);
          tolerance -= std::abs(eigenvalues[i]);
        }

        // Line 9 of streaming STHOSVD update
        // project orthgonal complement to new basis
        // K = E x_n V.T
        tensor_t K = ttm(E, n, V, true);

        // Line 10 of streaming STHOSVD update
        // pad core with zeros
        G = factorization.isvd.padTensorAlongMode(G, n, R_new);

        // Line 11 of streaming STHOSVD update
        // pad ISVD right singular vectors with zeros
        factorization.isvd.padRightSingularVectorsAlongMode(n, R_new);

        // Line 12 of streaming STHOSVD algorithm
        // prepare slice for next mode
        Y = factorization.isvd.concatenateTensorsAlongMode(D, K, n);

        // Line 13 of streaming STHOSVD algorithm
        // update basis
        const int N = U[n].extent(0);
        const int R = U[n].extent(1);
        matrix_t U_new("U_new", N, R+R_new);
        Kokkos::deep_copy(Kokkos::subview(U_new, Kokkos::ALL, std::make_pair(0,R)), U[n]);
        Kokkos::deep_copy(Kokkos::subview(U_new, Kokkos::ALL, std::make_pair(R,R+R_new)), V);
        U[n] = U_new;
      }
    }

    // Line 16 of streaming STHOSVD update algorithm
    // Add new row to ISVD factorization
    const scalar_t delta = std::sqrt(tolerance / Y.frobeniusNormSquared());
    factorization.isvd.updateFactorsWithNewSlice(Y, delta);

    factorization.squared_errors[ndims - 1] = std::pow(factorization.isvd.getErrorNorm(), 2);

    // Lines 17 of streaming STHOSVD update algorithm
    // Retrieve updated left singular vectors from ISVD factorization
    matrix_t U_new = Kokkos::create_mirror(factorization.isvd.getLeftSingularVectors());
    Kokkos::deep_copy(U_new, factorization.isvd.getLeftSingularVectors());

    // Line 22 of StreamingTuckerUpdate algorithm
    // Split U_new into two submatrices
    // Use first submatrix to update core
    {
      const int m_new = U_new.extent(0);
      const int r_new = U_new.extent(1);

      const int m_old = m_new - 1;
      const int r_old = U[ndims - 1].extent(1);

      auto U1 = Kokkos::subview(U_new, std::make_pair(0,m_old), Kokkos::ALL);
      matrix_t M("M", r_new, r_old);
      KokkosBlas::gemm("T", "N", scalar_t(1.0), U1, U[ndims-1], scalar_t(0.0), M);
      G = ttm(G, ndims - 1, M, false);
    }

    // Lines 23-24 of StreamingTuckerUpdate algorithm
    // Use last row of Unew[d] to update scale new slice and update core in-place
    {
      const int nrow = Y.size();
      const int ncol = U_new.extent(0);
      const int rank = U_new.extent(1);
      auto Yd = Y.data();
      auto Gd = G.data();
      Kokkos::parallel_for("Core Update",
                           Kokkos::RangePolicy<exec_space>(0,nrow),
                           KOKKOS_LAMBDA(const int i)
      {
        for (int j = 0; j < rank; ++j)
          Gd[i+j*nrow] += Yd[i]*U_new(ncol-1,j);
      });
    }

    // Line 25 of StreamingTuckerUpdate algorithm
    // Save new factor matrix
    U[ndims - 1] = U_new;

    // print status after streaming ST-HOSVD update
    {
      for (int n = 0; n < ndims; ++n) {
        log_stream << std::setw(6) << U[n].extent(0) << " " << std::flush;
      }
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization.Xnorm2) << " " << std::flush;
      for (int n = 0; n < ndims; ++n) {
        log_stream << std::setw(6) << U[n].extent(1) << " " << std::flush;
      }
      scalar_t Enorm2 = 0;
      for (int n = 0; n < ndims; ++n) {
        Enorm2 += factorization.squared_errors[n];
        log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization.squared_errors[n]) << " " << std::flush;
      }
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2) << " " << std::flush;
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2 / factorization.Xnorm2) << std::endl;
    }
  }

  // Close the file containing snapshot filenames
  inStream.close();

  // Save final factor matrices in factorization
  factorization.factorization = ttensor_t(G, U);

  return factorization;
}

// Explicit instantiations

//template class StreamingTuckerTensor<float>;
template class StreamingTuckerTensor<double>;

// template
// StreamingTuckerTensor<float,Kokkos::DefaultExecutionSpace::memory_space>
// StreamingSTHOSVD(const Tensor<float,Kokkos::DefaultExecutionSpace::memory_space>& X,
//                  const Tucker::TuckerTensor< Tensor<float,Kokkos::DefaultExecutionSpace::memory_space> >& initial_factorization,
//                  const TensorGramEigenvalues<float,Kokkos::DefaultExecutionSpace::memory_space>& initial_eigenvalues,
//                  const char* filename,
//                  const float epsilon,
//                  const std::string &streaming_stats_file,
//                  bool useQR,
//                  bool flipSign);

template
StreamingTuckerTensor<double,Kokkos::DefaultExecutionSpace::memory_space>
StreamingSTHOSVD(const Tensor<double,Kokkos::DefaultExecutionSpace::memory_space>& X,
                 const Tucker::TuckerTensor< Tensor<double,Kokkos::DefaultExecutionSpace::memory_space> >& initial_factorization,
                 const TensorGramEigenvalues<double,Kokkos::DefaultExecutionSpace::memory_space>& initial_eigenvalues,
                 const char* filename,
                 const double epsilon,
                 const std::string &streaming_stats_file,
                 bool useQR,
                 bool flipSign);

} // end namespace Tucker
