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
 * \brief Contains the functions for reading and writing tensors,
 * as well as the functions for computing the Gram matrix,
 * performing a tensor-times-matrix computation, and computing
 * a %Tucker decomposition
 *
 * @author Alicia Klinvex
 */

#ifndef TUCKER_MPI_HPP_
#define TUCKER_MPI_HPP_

#include "TuckerMPI_MPIWrapper.hpp"
#include "TuckerMPI_TuckerTensor.hpp"
#include "TuckerMPI_Matrix.hpp"
#include "Tucker_Metric.hpp"

/** \namespace TuckerMPI \brief Contains the data structures and
 * functions necessary for a parallel tucker decomposition
 */
namespace TuckerMPI {

/** \brief Computes the Gram matrix using the old algorithm
 *
 * \param Y A parallel tensor
 * \param n The dimension for the tensor unfolding
 * \param mult_timer Timer for the local multiplication
 * \param shift_timer Timer for the circular shift
 * \param allreduce_timer Timer for the all-reduce
 * \param allgather_timer Timer for the all-gather
 */
template <class scalar_t>
Tucker::Matrix<scalar_t>* oldGram(const Tensor<scalar_t>* Y, const int n,
    Tucker::Timer* mult_timer=0, Tucker::Timer* shift_timer=0,
    Tucker::Timer* allreduce_timer=0,
    Tucker::Timer* allgather_timer=0);

/** \brief Computes the Gram matrix using the new algorithm
 *
 * \param Y A parallel tensor
 * \param n The dimension for the tensor unfolding
 * \param mult_timer Timer for the local multiplication
 * \param pack_timer Timer for packing the data
 * \param alltoall_timer Timer for the all-to-all communication
 * \param unpack_timer Timer for unpacking the data
 * \param allreduce_timer Timer for the all-reduce
 */
template <class scalar_t>
Tucker::Matrix<scalar_t>* newGram(const Tensor<scalar_t>* Y, const int n,
    Tucker::Timer* mult_timer=0, Tucker::Timer* pack_timer=0,
    Tucker::Timer* alltoall_timer=0, Tucker::Timer* unpack_timer=0,
    Tucker::Timer* allreduce_timer=0);

/** \brief Computes a Tucker decomposition
 *
 * \param X A parallel tensor
 * \param epsilon Determines how many eigenvectors are discarded
 * \param useOldGram Determines whether the old Gram algorithm
 * is used, or the new one
 * \param flipSign If enabled, it flips the sign to be
 * comparable to the Matlab tensor toolbox results, where
 * the maximum magnitude entry in each vector of the factors U_n
 * is forced to be positive
 *
 * \todo There are no tests for this function yet.
 */
template <class scalar_t>
const TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* const X,
    const scalar_t epsilon, bool useOldGram=true,
    bool flipSign=false);

/** \brief Computes a Tucker decomposition
 *
 * \param X A parallel tensor
 * \param reducedI Dimensions of the desired core tensor
 * \param useOldGram Determines whether the old Gram algorithm
 * is used, or the new one
 * \param flipSign If enabled, it flips the sign to be
 * comparable to the Matlab tensor toolbox results, where
 * the maximum magnitude entry in each vector of the factors U_n
 * is forced to be positive
 *
 * \todo There are no tests for this function yet.
 */
template <class scalar_t>
const TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* const X,
    const Tucker::SizeArray* const reducedI, bool useOldGram=true,
    bool flipSign=false);

/** \brief Compute some information about slices of a tensor
 *
 * \param Y The tensor whose information is being computed
 * \param mode The mode used to determine the slices
 * \param metrics A sum of #Tucker::Metric to be computed
 */
template <class scalar_t>
Tucker::MetricData<scalar_t>* computeSliceMetrics(const Tensor<scalar_t>* const Y,
    int mode, int metrics);

/** \brief Perform a transformation on each slice of a tensor
 *
 * If slice \a mode is denoted as S, the transformation is as follows
 * S = (S + \a shifts[\a mode]) / \a scales[\a mode]
 *
 * \param Y The tensor whose slices are being transformed
 * \param mode The mode which determines the slices
 * \param scales Array of numbers to divide by
 * \param shifts Array of numbers to add
 */
template <class scalar_t>
void transformSlices(Tensor<scalar_t>* Y, int mode, const scalar_t* scales,
    const scalar_t* shifts);

/** \brief Normalize each slice of the tensor so its data lies in the range [0,1]
 *
 * \param Y The tensor whose slices are being normalized
 * \param mode The mode which determines the slices
 * \param stdThresh If the standard deviation is less than this value, set it to 1
 */
template <class scalar_t>
void normalizeTensorStandardCentering(Tensor<scalar_t>* Y, int mode, scalar_t stdThresh=1e-9);

template <class scalar_t>
void normalizeTensorMinMax(Tensor<scalar_t>* Y, int mode);

template <class scalar_t>
void normalizeTensorMax(Tensor<scalar_t>* Y, int mode);

template <class scalar_t>
const Tensor<scalar_t>* reconstructSingleSlice(const TuckerTensor<scalar_t>* fact,
    const int mode, const int sliceNum);

template <class scalar_t>
void readTensorBinary(std::string& filename, Tensor<scalar_t>& Y);

/** \brief Imports a parallel tensor using MPI_IO
 *
 * \param filename Binary file to be read
 * \param Y Parallel tensor to store the data.  This function will
 * not allocate any memory for Y; it will only change the values.
 *
 * \warning This function will crash if any process owns nothing
 */
template <class scalar_t>
void importTensorBinary(const char* filename, Tensor<scalar_t>* Y);

/** \brief Imports a sequential tensor using MPI_IO
 *
 * \param filename Binary file to be read
 * \param Y Sequential tensor to store the data.  This function will
 * not allocate any memory for Y; it will only change the values.
 */
template <class scalar_t>
void importTensorBinary(const char* filename, Tucker::Tensor<scalar_t>* Y);

template <class scalar_t>
void importTimeSeries(const char* filename, Tensor<scalar_t>* Y);

template <class scalar_t>
void writeTensorBinary(std::string& filename, const Tensor<scalar_t>& Y);

/** \brief Exports a parallel tensor using MPI_IO
 *
 * \param filename Binary file to be written to
 * \param Y Parallel tensor to be written to a file
 */
template <class scalar_t>
void exportTensorBinary(const char* filename, const Tensor<scalar_t>* Y);

/** \brief Exports a sequential tensor using MPI_IO
 *
 * \param filename Binary file to be written to
 * \param Y Sequential tensor to be written to a file
 */
template <class scalar_t>
void exportTensorBinary(const char* filename, const Tucker::Tensor<scalar_t>* Y);

template <class scalar_t>
void exportTimeSeries(const char* filename, const Tensor<scalar_t>* Y);

} // end namespace TuckerMPI

#endif /* TUCKER_MPI_HPP_ */
