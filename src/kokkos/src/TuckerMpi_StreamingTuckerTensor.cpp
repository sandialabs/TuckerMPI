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
#include <vector>
#include <numeric>

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas3_gemm.hpp>

#include "TuckerMpi.hpp"
#include "TuckerMpi_StreamingTuckerTensor.hpp"
#include "Tucker_create_mirror.hpp"
#include "TuckerOnNode_transform_slices.hpp"

/** \namespace Tucker \brief Contains the data structures and functions
 * necessary for a sequential tucker decomposition
 */
namespace TuckerMpi {

template <class scalar_t, class mem_space_t>
StreamingTuckerTensor<scalar_t,mem_space_t>
StreamingSTHOSVD(
  const Tensor<scalar_t,mem_space_t>& X,
  const Tucker::TuckerTensor< Tensor<scalar_t,mem_space_t> >& initial_factorization,
  const TuckerOnNode::TensorGramEigenvalues<scalar_t,mem_space_t>& initial_eigenvalues,
  const int scale_mode,
  const Kokkos::View<scalar_t*, mem_space_t>& scales,
  const Kokkos::View<scalar_t*, mem_space_t>& shifts,
  const char* filename,
  const scalar_t epsilon,
  Tucker::Timer &readTimer,
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
    auto eigs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initial_eigenvalues[n]);
    for (int i = G.globalExtent(n); i < U[n].extent(0); ++i) {
      if (eigs[i]) {
        factorization.squared_errors[n] += std::abs(eigs[i]);
      } else {
        // TODO - encountered null eigenvalues and null singular values - throw exception?
      }
    }
  }
  factorization.squared_errors[ndims - 1] = std::pow(factorization.isvd.getErrorNorm(), 2);

  const MPI_Comm& comm = G.getDistribution().getComm();
  int globalRank;
  MPI_Comm_rank(comm, &globalRank);

  std::ofstream log_stream;

  // print status header
  if (globalRank == 0) {
    log_stream.open(streaming_stats_file);
    log_stream << std::scientific;
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
  if (globalRank == 0) {
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
  std::vector<int> slice_dims = X.getDistribution().getGlobalDims();
  std::vector<int> slice_procs = X.getDistribution().getProcessorGrid().getSizeArray();
  slice_dims.pop_back();
  slice_procs.pop_back();

  // Create distribution to reuse for each slice, so we don't create a new
  // one each step
  Distribution slice_dist(slice_dims, slice_procs);

  //Open the file containing names of stream of snapshot files
  //Loop over, read each snapshot, update the Tucker model
  std::ifstream inStream(filename);
  std::string snapshot_file;

  if (globalRank == 0) {
    std::cout << std::endl
              << "---------------------------------------------\n"
              << "----- Streaming ST-HOSVD Starting" << " -----\n"
              << "---------------------------------------------\n"
              << std::flush;
  }

  while(inStream >> snapshot_file) {
    Tucker::Timer sthosvd_timer, ttm_timer, gram_timer, eig_timer, pad_timer,
      basis_update_timer, isvd_sv_update_timer, isvd_core_update_timer,
      local_read_timer;

    // Read the new tensor slice
    if (globalRank == 0)
      std::cout<< "Reading snapshot " << snapshot_file << std::endl;
    readTimer.start();
    local_read_timer.start();
    tensor_t Y(slice_dist);
    read_tensor_binary(globalRank, Y, snapshot_file);
    local_read_timer.stop();
    readTimer.stop();

    sthosvd_timer.start();

    // apply normalization if provided
    if (scales.extent(0) > 0 && shifts.extent(0) > 0)
      TuckerOnNode::transform_slices(
        Y.localTensor(), scale_mode, scales, shifts);

    // compute/update data norms
    const scalar_t Ynorm2 = Y.frobeniusNormSquared();
    factorization.Xnorm2 += Ynorm2;

    // compute total allowed error based off current error
    scalar_t tolerance = 0;
    {
      for (int n = 0; n < ndims; ++n) {
        assert(!std::isnan(factorization.squared_errors[n]));
        tolerance += factorization.squared_errors[n];
      }

      tolerance = epsilon * epsilon * factorization.Xnorm2 - tolerance;
    }

    // compute allowed error for non-streaming modes
    const scalar_t thresh = tolerance / ndims;

    // Loop over non-streaming modes
    for(int n=0; n<ndims-1; n++) {
#ifndef NDEBUG
      // PRINT_DEBUG
      printf("Y.globalSize(%d) = %d, Y.localSize(%d) = %d, Ynorm2 = %e, U[%d].extent(0) = %ld\n",
             n, Y.getDistribution().getGlobalDims()[n],
             n, Y.getDistribution().getLocalDims()[n],
             Y.frobeniusNormSquared(),
             n, U[n].extent(0));
#endif

      // Line 2 of streaming STHOSVD update
      // compute projection of new slice onto existing basis
      // D = Y x_n U[n].T
      // D must have a compatible distribution with V_
      tensor_t V_ = factorization.isvd.getRightSingularVectors();
      assert(U[n].extent(1) == V_.globalExtent(n));
      Distribution D_dist =
        Y.getDistribution().replaceModeWithSizes(n, U[n].extent(1),
                                                 V_.localExtent(n));
      ttm_timer.start();
      tensor_t D = ttm(Y, n, U[n], true, D_dist);
      ttm_timer.stop();

#ifndef NDEBUG
      // PRINT_DEBUG
      printf("D.globalSize(%d) = %d, D.localSize(%d) = %d, Dnorm2 = %e, U[%d].extent(1) = %ld\n",
             n, D.getDistribution().getGlobalDims()[n],
             n, D.getDistribution().getLocalDims()[n],
             D.frobeniusNormSquared(),
             n, U[n].extent(1));
#endif

      // Line 3 of streaming STHOSVD update
      // compute orthogonal complement of new slice w.r.t. existing basis
      // E = D x_n U[n]
      ttm_timer.start();
      tensor_t E = ttm(D, n, U[n], false);
      ttm_timer.stop();

#ifndef NDEBUG
      // PRINT_DEBUG
      printf("E.globalSize(%d) = %d, E.localSize(%d) = %d, Enorm2 = %e\n",
             n, E.getDistribution().getGlobalDims()[n],
             n, E.getDistribution().getLocalDims()[n],
             E.frobeniusNormSquared());
#endif

      // E = Y-E
      assert(E.getDistribution() == Y.getDistribution());
      KokkosBlas::axpby(scalar_t(1.0), Y.localTensor().data(), scalar_t(-1.0),
                        E.localTensor().data());

      const scalar_t Enorm2 = E.frobeniusNormSquared();

      if (Enorm2 <= thresh) {
        factorization.squared_errors[n] += Enorm2;
        assert(!std::isnan(factorization.squared_errors[n]));
        tolerance -= Enorm2;

        Y = D;
      } else {
        // Line 4 of streaming STHOSVD update
        // compute Gram of orthogonal complement of new slice
        gram_timer.start();
        matrix_t gram = compute_gram(E, n);
        gram_timer.stop();

        // Lines 5-8 of streaming STHOSVD update
        // compute truncated eigendecomposition of Gram
        eig_timer.start();
        auto eigenvalues_d = Tucker::impl::compute_and_sort_descending_eigvals_and_eigvecs_inplace(gram, false);
        auto eigenvalues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigenvalues_d);
        int nev = Tucker::impl::count_eigvals_using_threshold(eigenvalues, thresh);
        matrix_t V = Kokkos::subview(gram, Kokkos::ALL, std::make_pair(0, nev));
        eig_timer.stop();

        const int R_new = V.extent(1);
        for (int i=R_new; i<V.extent(0); ++i) {
          factorization.squared_errors[n] += std::abs(eigenvalues[i]);
          assert(!std::isnan(factorization.squared_errors[n]));
          tolerance -= std::abs(eigenvalues[i]);
        }

        // Line 9 of streaming STHOSVD update
        // project orthgonal complement to new basis
        // K = E x_n V.T
        ttm_timer.start();
        tensor_t K = ttm(E, n, V, true);
        assert(!K.localTensor().isNan());
        ttm_timer.stop();

        // Number of rows to add on my proc
        const int my_R = D.localExtent(n);
        const int my_R_new = K.localExtent(n);

        // Line 10 of streaming STHOSVD update
        // pad core with zeros
        pad_timer.start();
        G = factorization.isvd.padTensorAlongMode(G, n, my_R_new);
        assert(!G.localTensor().isNan());

        // Line 11 of streaming STHOSVD update
        // pad ISVD right singular vectors with zeros
        factorization.isvd.padRightSingularVectorsAlongMode(n, my_R_new);

        // Line 12 of streaming STHOSVD algorithm
        // prepare slice for next mode
        Y = factorization.isvd.concatenateTensorsAlongMode(D, K, n);
        assert(!Y.localTensor().isNan());
        pad_timer.stop();

        // Line 13 of streaming STHOSVD algorithm
        // update basis
        basis_update_timer.start();
        const int N = U[n].extent(0);
        const int R = U[n].extent(1);
        matrix_t U_new("U_new", N, R+R_new);
        const MPI_Comm& comm = slice_dist.getProcessorGrid().getColComm(n);
        const int num_proc = slice_dist.getProcessorGrid().getNumProcs(n);
        std::vector<int> R_proc(num_proc), R_new_proc(num_proc);
        MPI_Allgather(
          &my_R,     1, MPI_INT, R_proc.data(),     1, MPI_INT, comm);
        MPI_Allgather(
          &my_R_new, 1, MPI_INT, R_new_proc.data(), 1, MPI_INT, comm);
        assert(std::accumulate(R_proc.begin(), R_proc.end(), 0) == R);
        assert(std::accumulate(R_new_proc.begin(), R_new_proc.end(), 0) == R_new);
        int offset_U = 0;
        int offset_V = 0;
        int offset_U_new = 0;
        for (int k=0; k<num_proc; ++k) {
          auto U1 = Kokkos::subview(
            U[n], Kokkos::ALL, std::make_pair(offset_U,offset_U+R_proc[k]));
          auto U2 = Kokkos::subview(
            U_new, Kokkos::ALL, std::make_pair(offset_U_new,offset_U_new+R_proc[k]));
          Kokkos::deep_copy(U2, U1);
          offset_U += R_proc[k];
          offset_U_new += R_proc[k];
          U2 = Kokkos::subview(
            U_new, Kokkos::ALL, std::make_pair(offset_U_new,offset_U_new+R_new_proc[k]));
          auto V1 = Kokkos::subview(
            V, Kokkos::ALL, std::make_pair(offset_V,offset_V+R_new_proc[k]));
          Kokkos::deep_copy(U2, V1);
          offset_V += R_new_proc[k];
          offset_U_new += R_new_proc[k];
        }
        assert(offset_U == R);
        assert(offset_V == R_new);
        assert(offset_U_new == R+R_new);
        U[n] = U_new;
        basis_update_timer.stop();
      }
    }

    // Line 16 of streaming STHOSVD update algorithm
    // Add new row to ISVD factorization
    isvd_sv_update_timer.start();
    const scalar_t delta = std::sqrt(tolerance / Y.frobeniusNormSquared());
    factorization.isvd.updateFactorsWithNewSlice(Y, delta);
    isvd_sv_update_timer.stop();

    factorization.squared_errors[ndims - 1] = std::pow(factorization.isvd.getErrorNorm(), 2);
    assert(!std::isnan(factorization.squared_errors[ndims-1]));

    // Lines 17 of streaming STHOSVD update algorithm
    // Retrieve updated left singular vectors from ISVD factorization
    matrix_t U_new =
      Kokkos::create_mirror(mem_space_t(), factorization.isvd.getLeftSingularVectors());
    Kokkos::deep_copy(U_new, factorization.isvd.getLeftSingularVectors());

    // Line 22 of StreamingTuckerUpdate algorithm
    // Split U_new into two submatrices
    // Use first submatrix to update core
    isvd_core_update_timer.start();
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
      const int nrow = Y.localTensor().size(); // CHECK THIS
      const int ncol = U_new.extent(0);
      const int rank = U_new.extent(1);
      auto Yd = Y.localTensor().data();
      auto Gd = G.localTensor().data();
      Kokkos::parallel_for("Core Update",
                           Kokkos::RangePolicy<exec_space>(0,nrow),
                           KOKKOS_LAMBDA(const int i)
      {
        for (int j = 0; j < rank; ++j)
          Gd[i+j*nrow] += Yd[i]*U_new(ncol-1,j);
      });
    }
    isvd_core_update_timer.stop();

    // Line 25 of StreamingTuckerUpdate algorithm
    // Save new factor matrix
    U[ndims - 1] = U_new;

    sthosvd_timer.stop();

    if (globalRank == 0) {
      std::cout << "  Updated Tucker ranks:  ";
      for (int n = 0; n < ndims; ++n)
        std::cout << U[n].extent(1) << " ";
      std::cout << std::endl
                << "  Read time: " << local_read_timer.duration() << "s\n"
                << "  ST-HOSVD time: " << sthosvd_timer.duration() << "s\n"
                << "    Total TTM time: " << ttm_timer.duration() << "s\n"
                << "    Gram time: " << gram_timer.duration() << "s\n"
                << "    Eigen time: " << eig_timer.duration() << "s\n"
                << "    Tensor pad/concatenation time: " << pad_timer.duration() << "s\n"
                << "    Non-temporal basis update time: " << basis_update_timer.duration() << "s\n"
                << "    Temporal basis update time: " << isvd_sv_update_timer.duration() << "s\n"
                << "    Core update time: " << isvd_core_update_timer.duration() << "s\n"
                << std::flush;
    }

    // print status after streaming ST-HOSVD update
    if (globalRank == 0) {
      for (int n = 0; n < ndims; ++n) {
        log_stream << std::setw(6) << U[n].extent(0) << " " << std::flush;
      }
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization.Xnorm2) << " " << std::flush;
      for (int n = 0; n < ndims; ++n) {
        log_stream << std::setw(6) << U[n].extent(1) << " " << std::flush;
      }
      scalar_t Enorm2 = 0;
      for (int n = 0; n < ndims; ++n) {
        assert(!std::isnan(factorization.squared_errors[n]));
        Enorm2 += factorization.squared_errors[n];
        log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(factorization.squared_errors[n]) << " " << std::flush;
      }
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2) << " " << std::flush;
      log_stream << std::setw(12) << std::setprecision(6) << std::sqrt(Enorm2 / factorization.Xnorm2) << std::endl;
    }
  }

  if (globalRank == 0) {
    std::cout << "---------------------------------------------\n"
              << "----- Streaming ST-HOSVD Complete" << " -----\n"
              << "---------------------------------------------\n"
              << std::flush;
  }

  // Close the file containing snapshot filenames
  inStream.close();

  // Close the logging sream
  if (globalRank == 0)
    log_stream.close();

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
//                  const int scale_mode,
//                  const Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space>& scales,
//                  const Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space>& shifts,
//                  const char* filename,
//                  const float epsilon,
//                  Tucker::Timer &,
//                  const std::string &streaming_stats_file,
//                  bool useQR,
//                  bool flipSign);

template
StreamingTuckerTensor<double,Kokkos::DefaultExecutionSpace::memory_space>
StreamingSTHOSVD(const Tensor<double,Kokkos::DefaultExecutionSpace::memory_space>& X,
                 const Tucker::TuckerTensor< Tensor<double,Kokkos::DefaultExecutionSpace::memory_space> >& initial_factorization,
                 const TuckerOnNode::TensorGramEigenvalues<double,Kokkos::DefaultExecutionSpace::memory_space>& initial_eigenvalues,
                 const int scale_mode,
                 const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>& scales,
                 const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>& shifts,
                 const char* filename,
                 const double epsilon,
                 Tucker::Timer &,
                 const std::string &streaming_stats_file,
                 bool useQR,
                 bool flipSign);

} // end namespace Tucker
