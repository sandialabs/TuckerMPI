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
 * \brief Contains the parallel Matrix class
 *
 * \author Alicia Klinvex
 */

#include "TuckerMPI_Matrix.hpp"

namespace TuckerMPI {

template <class scalar_t>
Matrix<scalar_t>::Matrix(int nrows, int ncols, const MPI_Comm& comm, bool isBlockRow) :
  globalRows_(nrows),
  globalCols_(ncols),
  comm_(comm)
{
  // Get the communicator information
  int rank, nprocs;
  MPI_Comm_rank(comm_,&rank);
  MPI_Comm_size(comm_,&nprocs);

  // Create a map
  if(isBlockRow) {
    map_ = Tucker::MemoryManager::safe_new<Map>(nrows,comm);
  }
  else {
    map_ = Tucker::MemoryManager::safe_new<Map>(ncols,comm);
  }

  // Get the local number of rows and columns
  int localRows, localCols;
  if(isBlockRow) {
    localRows = map_->getLocalNumEntries();
    localCols = ncols;
  }
  else {
    localRows = nrows;
    localCols = map_->getLocalNumEntries();
  }

  // Create the local portion of the matrix
  localMatrix_ = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(localRows,localCols);
}

template <class scalar_t>
Matrix<scalar_t>::~Matrix()
{
  Tucker::MemoryManager::safe_delete(localMatrix_);
  Tucker::MemoryManager::safe_delete(map_);
}

template <class scalar_t>
Tucker::Matrix<scalar_t>* Matrix<scalar_t>::getLocalMatrix()
{
  return localMatrix_;
}

template <class scalar_t>
const Tucker::Matrix<scalar_t>* Matrix<scalar_t>::getLocalMatrix() const
{
  return localMatrix_;
}

template <class scalar_t>
size_t Matrix<scalar_t>::getLocalNumEntries() const
{
  return localMatrix_->getNumElements();
}

template <class scalar_t>
int Matrix<scalar_t>::getGlobalNumRows() const
{
  return globalRows_;
}

template <class scalar_t>
int Matrix<scalar_t>::getLocalNumRows() const
{
  return localMatrix_->nrows();
}

template <class scalar_t>
int Matrix<scalar_t>::getGlobalNumCols() const
{
  return globalCols_;
}

template <class scalar_t>
int Matrix<scalar_t>::getLocalNumCols() const
{
  return localMatrix_->ncols();
}

template <class scalar_t>
const Map* Matrix<scalar_t>::getMap() const
{
  return map_;
}

template <class scalar_t>
void Matrix<scalar_t>::print() const
{
  localMatrix_->print();
}

// Explicit instantiations to build static library for both single and double precision
template class Matrix<float>;
template class Matrix<double>;

} /* namespace TuckerMPI */
