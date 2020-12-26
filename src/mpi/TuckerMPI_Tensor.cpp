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
 * \brief Contains a class for storing a distributed tensor
 *
 * @author Alicia Klinvex
 */

#include <limits>
#include "TuckerMPI_Tensor.hpp"

namespace TuckerMPI {

template <class scalar_t>
Tensor<scalar_t>::Tensor(const Distribution* dist) :
    dist_(dist)
{
  localTensor_ = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(dist->getLocalDims());
}

template <class scalar_t>
Tensor<scalar_t>::~Tensor()
{
  Tucker::MemoryManager::safe_delete(dist_);
  Tucker::MemoryManager::safe_delete(localTensor_);
}

template <class scalar_t>
Tucker::Tensor<scalar_t>* Tensor<scalar_t>::getLocalTensor()
{
  return localTensor_;
}

template <class scalar_t>
const Tucker::Tensor<scalar_t>* Tensor<scalar_t>::getLocalTensor() const
{
  return localTensor_;
}

template <class scalar_t>
const Tucker::SizeArray& Tensor<scalar_t>::getGlobalSize() const
{
  return dist_->getGlobalDims();
}

template <class scalar_t>
const Tucker::SizeArray& Tensor<scalar_t>::getLocalSize() const
{
  return dist_->getLocalDims();
}

template <class scalar_t>
int Tensor<scalar_t>::getGlobalSize(int n) const
{
  return dist_->getGlobalDims()[n];
}

template <class scalar_t>
int Tensor<scalar_t>::getLocalSize(int n) const
{
  return dist_->getLocalDims()[n];
}

template <class scalar_t>
int Tensor<scalar_t>::getNumDimensions() const
{
  return dist_->getGlobalDims().size();
}


// This function is necessary for TTM
// TTM needs to know how many entries EACH processor has,
// not just THIS processor.
// We don't technically need to return the entire distribution;
// that part is flexible.
template <class scalar_t>
const Distribution* Tensor<scalar_t>::getDistribution() const
{
  return dist_;
}

template <class scalar_t>
size_t Tensor<scalar_t>::getLocalNumEntries() const
{
  return localTensor_->getNumElements();
}

template <class scalar_t>
size_t Tensor<scalar_t>::getGlobalNumEntries() const
{
  return dist_->getGlobalDims().prod();
}


// Compute the norm squared
template <class scalar_t>
scalar_t Tensor<scalar_t>::norm2() const
{
  // Compute the local portion
  scalar_t localNorm2 = localTensor_->norm2();

  // Perform a reduction
  scalar_t globalNorm2;
  MPI_Allreduce_(&localNorm2, &globalNorm2, 1,
      MPI_SUM, MPI_COMM_WORLD);
  return globalNorm2;
}

template <class scalar_t>
void Tensor<scalar_t>::print() const
{
  localTensor_->print();
}

template <class scalar_t>
void Tensor<scalar_t>::rand()
{
  localTensor_->rand();
}

// \todo This function is never tested
template <class scalar_t>
Tensor<scalar_t>* Tensor<scalar_t>::subtract(const Tensor<scalar_t>* t) const
{
  Tensor* sub = Tucker::MemoryManager::safe_new<Tensor>(dist_);

  size_t nnz = getLocalNumEntries();
  if(nnz > 0) {
    scalar_t* subdata = sub->localTensor_->data();
    scalar_t* thisdata = localTensor_->data();
    scalar_t* tdata = t->localTensor_->data();

    for(size_t i=0; i<nnz; i++) {
      subdata[i] = thisdata[i] - tdata[i];
    }
  }
  return sub;
}

// \todo This function is never tested
template <class scalar_t>
scalar_t Tensor<scalar_t>::maxEntry() const
{
  scalar_t localMax = std::numeric_limits<scalar_t>::lowest();
  size_t nnz = getLocalNumEntries();
  if(nnz > 0) {
    scalar_t* data = localTensor_->data();
    for(size_t i=0; i<nnz; i++) {
      localMax = std::max(localMax,data[i]);
    }
  }

  scalar_t globalMax;
  MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE,
            MPI_MAX, MPI_COMM_WORLD);
  return globalMax;
}

// \todo This function is never tested
template <class scalar_t>
scalar_t Tensor<scalar_t>::minEntry() const
{
  scalar_t localMin = std::numeric_limits<scalar_t>::max();
  size_t nnz = getLocalNumEntries();
  if(nnz > 0) {
    scalar_t* data = localTensor_->data();
    for(size_t i=0; i<nnz; i++) {
      localMin = std::min(localMin,data[i]);
    }
  }

  scalar_t globalMin;
  MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE,
            MPI_MIN, MPI_COMM_WORLD);
  return globalMin;
}

template <class scalar_t>
bool isApproxEqual(const Tensor<scalar_t>* t1, const Tensor<scalar_t>* t2,
    scalar_t tol)
{
  if(t1->getGlobalSize() != t2->getGlobalSize()) {
    std::cerr << "t1 and t2 have different global sizes\n";
    std::cerr << "t1: " << t1->getGlobalSize() << std::endl;
    std::cerr << "t2: " << t2->getGlobalSize() << std::endl;
    return false;
  }
  return (isApproxEqual(t1->getLocalTensor(),
      t2->getLocalTensor(), tol));
}

// Explicit instantiations to build static library for both single and double precision
template class Tensor<float>;
template class Tensor<double>;

template bool isApproxEqual(const Tensor<float>*, const Tensor<float>*, float);
template bool isApproxEqual(const Tensor<double>*, const Tensor<double>*, double);

} /* namespace TuckerMPI */
