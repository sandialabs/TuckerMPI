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
 * \brief Stores a parallel %Tucker decomposition
 *
 * @author Alicia Klinvex
 */

#ifndef TUCKERTENSOR_MPI_HPP_
#define TUCKERTENSOR_MPI_HPP_

#include <fstream>
#include "Tucker_Timer.hpp"
#include "Tucker_Matrix.hpp"
#include "TuckerMPI_MPIWrapper.hpp"
#include "TuckerMPI_Tensor.hpp"
#include "TuckerMPI_ttm.hpp"

namespace TuckerMPI {

/** \brief A structure for storing a %Tucker decomposition
 *
 * It is essentially a struct (all data is public),
 * but with a constructor and destructor
 */
template <class scalar_t>
class TuckerTensor {
public:
  /** Constructor
   * \param numDims The number of dimensions
   */
  TuckerTensor(const int numDims)
  {
    N = numDims;
    U = Tucker::MemoryManager::safe_new_array<Tucker::Matrix<scalar_t>*>(N);
    eigenvalues = Tucker::MemoryManager::safe_new_array<scalar_t*>(N);
    singularValues = Tucker::MemoryManager::safe_new_array<scalar_t*>(N);
    for(int i=0; i<N; i++) {
      U[i] = 0;
      eigenvalues[i] = 0;
      singularValues[i] = 0;
    }
    G = 0;

    LQ_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    LQ_bcast_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    LQ_redistribute_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    LQ_tsqr_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    LQ_localqr_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);

    svd_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);

    gram_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    gram_matmul_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    gram_shift_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    gram_allreduce_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    gram_allgather_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    gram_pack_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    gram_alltoall_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    gram_unpack_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);

    eigen_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);

    ttm_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    ttm_matmul_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    ttm_pack_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    ttm_reducescatter_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
    ttm_reduce_timer_ = Tucker::MemoryManager::safe_new_array<Tucker::Timer>(numDims);
  }

  /** Destructor */
  ~TuckerTensor()
  {
    if(G) Tucker::MemoryManager::safe_delete(G);
    for(int i=0; i<N; i++) {
      if(eigenvalues[i]) Tucker::MemoryManager::safe_delete_array(eigenvalues[i],U[i]->nrows());
      if(U[i]) Tucker::MemoryManager::safe_delete(U[i]);
      if(singularValues[i]) Tucker::MemoryManager::safe_delete_array(singularValues[i],U[i]->nrows());
    }
    Tucker::MemoryManager::safe_delete_array<Tucker::Matrix<scalar_t>*>(U,N);
    Tucker::MemoryManager::safe_delete_array<scalar_t*>(eigenvalues,N);
    Tucker::MemoryManager::safe_delete_array<scalar_t*>(singularValues,N);

    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_matmul_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_shift_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_allreduce_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_allgather_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_pack_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_alltoall_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(gram_unpack_timer_,N);

    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(eigen_timer_,N);

    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(LQ_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(LQ_redistribute_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(LQ_tsqr_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(LQ_localqr_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(LQ_bcast_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(svd_timer_,N);

    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(ttm_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(ttm_matmul_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(ttm_pack_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(ttm_reducescatter_timer_,N);
    Tucker::MemoryManager::safe_delete_array<Tucker::Timer>(ttm_reduce_timer_,N);
  }

  Tensor<scalar_t>* reconstructTensor() const
  {
    Tensor<scalar_t>* temp = G;
    for(int mode=0; mode<N; mode++) {
      Tensor<scalar_t>* t = ttm(temp,mode,U[mode]);

      // At iteration 0, temp = G
      if(mode > 0) {
        Tucker::MemoryManager::safe_delete(temp);
      }
      temp = t;
    }
    return temp;
  }

  /** \brief Prints some runtime information
   *
   * \todo This can be made more efficient
   */
  void printTimersLQ(const std::string& filename) const
  {
    const int ntimers = 11;
    scalar_t* raw_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);

    // Get the MPI data
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // Pack the data
    for(int i=0; i<N; i++) {
      raw_array[i*ntimers] = LQ_timer_[i].duration();
      raw_array[i*ntimers+1] = LQ_redistribute_timer_[i].duration();
      raw_array[i*ntimers+2] = LQ_localqr_timer_[i].duration();
      raw_array[i*ntimers+3] = LQ_tsqr_timer_[i].duration();
      raw_array[i*ntimers+4] = LQ_bcast_timer_[i].duration();

      raw_array[i*ntimers+5] = svd_timer_[i].duration();

      raw_array[i*ntimers+6] = ttm_timer_[i].duration();
      raw_array[i*ntimers+7] = ttm_matmul_timer_[i].duration();
      raw_array[i*ntimers+8] = ttm_pack_timer_[i].duration();
      raw_array[i*ntimers+9] = ttm_reducescatter_timer_[i].duration();
      raw_array[i*ntimers+10] = ttm_reduce_timer_[i].duration();
    }
    raw_array[ntimers*N] = total_timer_.duration();

    // Allocate memory on process 0
    scalar_t* gathered_data;
    scalar_t* min_array;
    scalar_t* max_array;
    scalar_t* mean_array;
    if(rank == 0) {
      min_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);
      max_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);
      mean_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);
      gathered_data = Tucker::MemoryManager::safe_new_array<scalar_t>((ntimers*N+1)*nprocs);
    }
    else {
      min_array = 0;
      max_array = 0;
      mean_array = 0;
      gathered_data = 0;
    }

    // Perform the reductions
    MPI_Reduce_(raw_array, min_array, ntimers*N+1,
        MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce_(raw_array, max_array, ntimers*N+1,
        MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce_(raw_array, mean_array, ntimers*N+1,
        MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather all the data to process 0
    MPI_Gather_(raw_array, ntimers*N+1,
        gathered_data, ntimers*N+1, 0, MPI_COMM_WORLD);

    if(rank == 0) {
      // mean_array currently holds the sum, so divide by # entries
      for(int i=0; i<ntimers*N+1; i++) {
        mean_array[i] /= nprocs;
      }

      std::cout << "\n\n           Timing results for " << nprocs << " MPI processes\n"
                << "Timer             Min          Max          Mean\n"
                << "--------------------------------------------------------\n";
      for(int i=0; i<N; i++) {
        std::cout << "LQ(" << i << ")         : " << std::scientific
          << min_array[i*ntimers] << " " << std::scientific
          << max_array[i*ntimers] << " " << std::scientific
          << mean_array[i*ntimers] << std::endl;

        std::cout << "redistribute(" << i << ")         : " << std::scientific
          << min_array[i*ntimers+1] << " " << std::scientific
          << max_array[i*ntimers+1] << " " << std::scientific
          << mean_array[i*ntimers+1] << std::endl;

        std::cout << "local qr(" << i << ")         : " << std::scientific
          << min_array[i*ntimers+2] << " " << std::scientific
          << max_array[i*ntimers+2] << " " << std::scientific
          << mean_array[i*ntimers+2] << std::endl;

        std::cout << "TSQR(" << i << ")         : " << std::scientific
          << min_array[i*ntimers+3] << " " << std::scientific
          << max_array[i*ntimers+3] << " " << std::scientific
          << mean_array[i*ntimers+3] << std::endl;

        std::cout << "bcast(" << i << ")         : " << std::scientific
          << min_array[i*ntimers+4] << " " << std::scientific
          << max_array[i*ntimers+4] << " " << std::scientific
          << mean_array[i*ntimers+4] << std::endl;

        std::cout << "SVD(" << i << ")         : " << std::scientific
          << min_array[i*ntimers+5] << " " << std::scientific
          << max_array[i*ntimers+5] << " " << std::scientific
          << mean_array[i*ntimers+5] << std::endl;

        std::cout << "TTM(" << i << ")         : " << std::scientific
          << min_array[i*ntimers+6] << " " << std::scientific
          << max_array[i*ntimers+6] << " " << std::scientific
          << mean_array[i*ntimers+6] << std::endl;

        std::cout << " local multiply : " << std::scientific
          << min_array[i*ntimers+7] << " " << std::scientific
          << max_array[i*ntimers+7] << " " << std::scientific
          << mean_array[i*ntimers+7] << std::endl;

        std::cout << " packing        : " << std::scientific
          << min_array[i*ntimers+8] << " " << std::scientific
          << max_array[i*ntimers+8] << " " << std::scientific
          << mean_array[i*ntimers+8] << std::endl;

        std::cout << " reduce-scatter         : " << std::scientific
          << min_array[i*ntimers+9] << " " << std::scientific
          << max_array[i*ntimers+9] << " " << std::scientific
          << mean_array[i*ntimers+9] << std::endl;

        std::cout << " reduce         : " << std::scientific
          << min_array[i*ntimers+10] << " " << std::scientific
          << max_array[i*ntimers+10] << " " << std::scientific
          << mean_array[i*ntimers+10] << std::endl;
        

        std::cout << std::endl;
      }

      std::cout << "Total           : " << std::scientific
          << min_array[ntimers*N] << " " << std::scientific
          << max_array[ntimers*N] << " " << std::scientific
          << mean_array[ntimers*N] << std::endl << std::endl;

      // Send the data to a file
      std::ofstream os(filename);

      // Create the header row
      for(int d=0; d<N; d++) {
        os << "LQ(" << d << "),redistribute(" << d << "),localqr(" << d 
            << "),tsqr(" << d <<"),bcast(" << d << "),svd(" << d
            << "),TTM(" << d << "),TTM local multiply(" << d
            << "),TTM packing(" << d << "),TTM reduce-scatter(" << d
            << "),TTM reduce(" << d << "),";
      }
      os << "Total\n";

      // For each MPI process
      for(int r=0; r<nprocs; r++) {
        // For each timer belonging to that process
        for(int t=0; t<ntimers*N; t++) {
          os << gathered_data[r*(ntimers*N+1)+t] << ",";
        }
        os << gathered_data[r*(ntimers*N+1)+ntimers*N] << std::endl;
      }

      os.close();

      Tucker::MemoryManager::safe_delete_array<scalar_t>(min_array,ntimers*N+1);
      Tucker::MemoryManager::safe_delete_array<scalar_t>(max_array,ntimers*N+1);
      Tucker::MemoryManager::safe_delete_array<scalar_t>(mean_array,ntimers*N+1);
    }

    Tucker::MemoryManager::safe_delete_array<scalar_t>(raw_array,ntimers*N+1);
  }

  /** \brief Prints some runtime information
   *
   * \todo This can be made more efficient
   */
  void printTimers(const std::string& filename) const
  {
    const int ntimers = 14;
    scalar_t* raw_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);

    // Get the MPI data
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // Pack the data
    for(int i=0; i<N; i++) {
      raw_array[i*ntimers] = gram_timer_[i].duration();
      raw_array[i*ntimers+1] = gram_matmul_timer_[i].duration();
      raw_array[i*ntimers+2] = gram_shift_timer_[i].duration();
      raw_array[i*ntimers+3] = gram_allreduce_timer_[i].duration();
      raw_array[i*ntimers+4] = gram_allgather_timer_[i].duration();
      raw_array[i*ntimers+5] = gram_pack_timer_[i].duration();
      raw_array[i*ntimers+6] = gram_alltoall_timer_[i].duration();
      raw_array[i*ntimers+7] = gram_unpack_timer_[i].duration();

      raw_array[i*ntimers+8] = eigen_timer_[i].duration();

      raw_array[i*ntimers+9] = ttm_timer_[i].duration();
      raw_array[i*ntimers+10] = ttm_matmul_timer_[i].duration();
      raw_array[i*ntimers+11] = ttm_pack_timer_[i].duration();
      raw_array[i*ntimers+12] = ttm_reducescatter_timer_[i].duration();
      raw_array[i*ntimers+13] = ttm_reduce_timer_[i].duration();
    }
    raw_array[ntimers*N] = total_timer_.duration();

    // Allocate memory on process 0
    scalar_t* gathered_data;
    scalar_t* min_array;
    scalar_t* max_array;
    scalar_t* mean_array;
    if(rank == 0) {
      min_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);
      max_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);
      mean_array = Tucker::MemoryManager::safe_new_array<scalar_t>(ntimers*N+1);
      gathered_data = Tucker::MemoryManager::safe_new_array<scalar_t>((ntimers*N+1)*nprocs);
    }
    else {
      min_array = 0;
      max_array = 0;
      mean_array = 0;
      gathered_data = 0;
    }

    // Perform the reductions
    MPI_Reduce_(raw_array, min_array, ntimers*N+1,
        MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce_(raw_array, max_array, ntimers*N+1,
        MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce_(raw_array, mean_array, ntimers*N+1,
        MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather all the data to process 0
    MPI_Gather_(raw_array, ntimers*N+1, 
        gathered_data, ntimers*N+1, 0, MPI_COMM_WORLD);

    if(rank == 0) {
      // mean_array currently holds the sum, so divide by # entries
      for(int i=0; i<ntimers*N+1; i++) {
        mean_array[i] /= nprocs;
      }

      std::cout << "\n\n           Timing results for " << nprocs << " MPI processes\n"
                << "Timer             Min          Max          Mean\n"
                << "--------------------------------------------------------\n";
      for(int i=0; i<N; i++) {
        std::cout << "Gram(" << i << ")         : " << std::scientific
            << min_array[i*ntimers] << " " << std::scientific
            << max_array[i*ntimers] << " " << std::scientific
            << mean_array[i*ntimers] << std::endl;

        std::cout << " local multiply : " << std::scientific
            << min_array[i*ntimers+1] << " " << std::scientific
            << max_array[i*ntimers+1] << " " << std::scientific
            << mean_array[i*ntimers+1] << std::endl;

        if(max_array[i*ntimers+2] > 0) {
          std::cout << " shift          : " << std::scientific
              << min_array[i*ntimers+2] << " " << std::scientific
              << max_array[i*ntimers+2] << " " << std::scientific
              << mean_array[i*ntimers+2] << std::endl;
        }

        if(max_array[i*ntimers+3] > 0) {
          std::cout << " all-reduce     : " << std::scientific
              << min_array[i*ntimers+3] << " " << std::scientific
              << max_array[i*ntimers+3] << " " << std::scientific
              << mean_array[i*ntimers+3] << std::endl;
        }

        if(max_array[i*ntimers+4] > 0) {
          std::cout << " all-gather     : " << std::scientific
              << min_array[i*ntimers+4] << " " << std::scientific
              << max_array[i*ntimers+4] << " " << std::scientific
              << mean_array[i*ntimers+4] << std::endl;
        }

        if(max_array[i*ntimers+5] > 0) {
          std::cout << " packing        : " << std::scientific
              << min_array[i*ntimers+5] << " " << std::scientific
              << max_array[i*ntimers+5] << " " << std::scientific
              << mean_array[i*ntimers+5] << std::endl;
        }

        if(max_array[i*ntimers+6] > 0) {
          std::cout << " all-to-all     : " << std::scientific
              << min_array[i*ntimers+6] << " " << std::scientific
              << max_array[i*ntimers+6] << " " << std::scientific
              << mean_array[i*ntimers+6] << std::endl;
        }

        if(max_array[i*ntimers+7] > 0) {
          std::cout << " unpacking      : " << std::scientific
              << min_array[i*ntimers+7] << " " << std::scientific
              << max_array[i*ntimers+7] << " " << std::scientific
              << mean_array[i*ntimers+7] << std::endl;
        }

        std::cout << "Eigensolve(" << i << ")   : " << std::scientific
            << min_array[i*ntimers+8] << " " << std::scientific
            << max_array[i*ntimers+8] << " " << std::scientific
            << mean_array[i*ntimers+8] << std::endl;

        std::cout << "TTM(" << i << ")          : " << std::scientific
            << min_array[i*ntimers+9] << " " << std::scientific
            << max_array[i*ntimers+9] << " " << std::scientific
            << mean_array[i*ntimers+9] << std::endl;

        std::cout << " local multiply : " << std::scientific
            << min_array[i*ntimers+10] << " " << std::scientific
            << max_array[i*ntimers+10] << " " << std::scientific
            << mean_array[i*ntimers+10] << std::endl;

        if(max_array[i*ntimers+11] > 0) {
          std::cout << " packing        : " << std::scientific
              << min_array[i*ntimers+11] << " " << std::scientific
              << max_array[i*ntimers+11] << " " << std::scientific
              << mean_array[i*ntimers+11] << std::endl;
        }

        if(max_array[i*ntimers+12] > 0) {
          std::cout << " reduce-scatter : " << std::scientific
              << min_array[i*ntimers+12] << " " << std::scientific
              << max_array[i*ntimers+12] << " " << std::scientific
              << mean_array[i*ntimers+12] << std::endl;
        }

        if(max_array[i*ntimers+13] > 0) {
          std::cout << " reduce         : " << std::scientific
              << min_array[i*ntimers+13] << " " << std::scientific
              << max_array[i*ntimers+13] << " " << std::scientific
              << mean_array[i*ntimers+13] << std::endl;
        }

        std::cout << std::endl;
      }

      std::cout << "Total           : " << std::scientific
          << min_array[ntimers*N] << " " << std::scientific
          << max_array[ntimers*N] << " " << std::scientific
          << mean_array[ntimers*N] << std::endl << std::endl;

      // Send the data to a file
      std::ofstream os(filename);

      // Create the header row
      for(int d=0; d<N; d++) {
        os << "Gram(" << d << "),Gram local multiply(" << d << "),Gram shift("
            << d << "),Gram all-reduce(" << d << "),Gram all-gather("
            << d << "),Gram packing(" << d << "),Gram all-to-all(" << d
            << "),Gram unpacking(" << d << "),Eigensolve(" << d
            << "),TTM(" << d << "),TTM local multiply(" << d
            << "),TTM packing(" << d << "),TTM reduce-scatter(" << d
            << "),TTM reduce(" << d << "),";
      }
      os << "Total\n";

      // For each MPI process
      for(int r=0; r<nprocs; r++) {
        // For each timer belonging to that process
        for(int t=0; t<ntimers*N; t++) {
          os << gathered_data[r*(ntimers*N+1)+t] << ",";
        }
        os << gathered_data[r*(ntimers*N+1)+ntimers*N] << std::endl;
      }

      os.close();

      Tucker::MemoryManager::safe_delete_array(min_array,ntimers*N+1);
      Tucker::MemoryManager::safe_delete_array(max_array,ntimers*N+1);
      Tucker::MemoryManager::safe_delete_array(mean_array,ntimers*N+1);
    }
    Tucker::MemoryManager::safe_delete_array(raw_array,ntimers*N+1);
  }

  Tensor<scalar_t>* G; //!< the tensor of reduced size
  Tucker::Matrix<scalar_t>** U; //!< an array of factors/dense matrices
  int N; //!< the number of factors
  scalar_t** eigenvalues; //!< the eigenvalues of each Gram matrix
  scalar_t** singularValues; // the singular values of L, same as that of the tensor unfolding.


  /// \brief Array of timers for LQ computation
  Tucker::Timer* LQ_timer_;

  /// \brief Array of timers for broadcasting L
  Tucker::Timer* LQ_bcast_timer_;

  /// \brief Array of timers for broadcasting L
  Tucker::Timer* LQ_redistribute_timer_;

  /// \brief Array of timers for broadcasting L
  Tucker::Timer* LQ_tsqr_timer_;

  /// \brief Array of timers for broadcasting L
  Tucker::Timer* LQ_localqr_timer_;


  /// \brief Array of timers for broadcasting L
  Tucker::Timer* svd_timer_;
    
  /// \brief Array of timers for Gram matrix computation
  Tucker::Timer* gram_timer_;

  /// \brief Array of timers for local matrix multiply within Gram
  Tucker::Timer* gram_matmul_timer_;

  /// \brief Array of timers for circular-shift within Gram
  Tucker::Timer* gram_shift_timer_;

  /// \brief Array of timers for all-reduce within Gram
  Tucker::Timer* gram_allreduce_timer_;

  /// \brief Array of timers for all-gather within Gram
  Tucker::Timer* gram_allgather_timer_;

  /// \brief Array of timers for pack within Gram
  Tucker::Timer* gram_pack_timer_;

  /// \brief Array of timers for all-to-all within Gram
  Tucker::Timer* gram_alltoall_timer_;

  /// \brief Array of timers for unpack within Gram
  Tucker::Timer* gram_unpack_timer_;

  /// \brief Array of timers for eigensolver computation
  Tucker::Timer* eigen_timer_;

  /// \brief Array of timers for TTM computation
  Tucker::Timer* ttm_timer_;

  /// \brief Array of timers for local matrix multiply within TTM
  Tucker::Timer* ttm_matmul_timer_;

  /// \brief Array of timers for pack within TTM
  Tucker::Timer* ttm_pack_timer_;

  /// \brief Array of timers for reduce-scatter within TTM
  Tucker::Timer* ttm_reducescatter_timer_;

  /// \brief Array of timers for reduce within TTM
  Tucker::Timer* ttm_reduce_timer_;

  /// \brief Total ST-HOSVD runtime
  Tucker::Timer total_timer_;

private:
  /// @cond EXCLUDE
  TuckerTensor(const TuckerTensor& tt);
  /// @endcond

  
};

// Explicit instantiations to build static library for both single and scalar_t precision
template class TuckerTensor<float>;
template class TuckerTensor<double>;

} // end namespace TuckerMPI

#endif /* TUCKERTENSOR_MPI_HPP_ */
