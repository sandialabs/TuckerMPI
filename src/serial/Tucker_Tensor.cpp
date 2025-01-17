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
 * \brief Defines a class for storing a tensor
 *
 * @author Alicia Klinvex
 */

#include <iostream>
#include <exception>
#include <cmath>
#include <cstdlib>
#include "Tucker_Tensor.hpp"
#include <iomanip> 

namespace Tucker {

template<class scalar_t>
Tensor<scalar_t>::Tensor(const SizeArray& I) :
    I_(I.size())
{
  // Copy the SizeArray
  for(int i=0; i<I.size(); i++) {
    if(I[i] < 0) {
      std::ostringstream oss;
      oss << "Tucker::Tensor::Tensor(const SizeArray& I): I["
          << i << "] = " << I[i] << " < 0.";
      throw std::length_error(oss.str());
    }
    I_[i] = I[i];
  }

  // Compute the total number of entries in this tensor
  size_t numEntries = getNumElements();
  if(numEntries > 0)
    data_ = MemoryManager::safe_new_array<scalar_t>(numEntries);
  else
    data_ = 0;
}

template<class scalar_t>
Tensor<scalar_t>::Tensor(int nrows, int ncols) :
    I_(2)
{
  I_[0] = nrows;
  I_[1] = ncols;

  if(nrows < 0) {
    std::ostringstream oss;
    oss << "Tucker::Tensor<scalar_t>::Tensor(int nrows, int ncols): nrows = "
        << nrows << " < 0";
    throw std::length_error(oss.str());
  }
  if(ncols < 0) {
    std::ostringstream oss;
    oss << "Tucker::Tensor<scalar_t>::Tensor(int nrows, int ncols): ncols = "
        << ncols << " < 0";
    throw std::length_error(oss.str());
  }

  size_t numEntries = nrows*ncols;
  if(numEntries > 0)
    data_ = MemoryManager::safe_new_array<scalar_t>(numEntries);
  else
    data_ = 0;
}

template<class scalar_t>
Tensor<scalar_t>::Tensor(int nrows) :
    I_(1)
{
  I_[0] = nrows;

  if(nrows < 0) {
    std::ostringstream oss;
    oss << "Tucker::Tensor<scalar_t>::Tensor(int nrows): nrows = "
        << nrows << " < 0";
    throw std::length_error(oss.str());
  }

  size_t numEntries = nrows;
  if(numEntries > 0)
    data_ = MemoryManager::safe_new_array<scalar_t>(numEntries);
  else
    data_ = 0;
}

template<class scalar_t>
Tensor<scalar_t>::~Tensor()
{
  if(data_)
    MemoryManager::safe_delete_array<scalar_t>(data_,getNumElements());
}

template<class scalar_t>
int Tensor<scalar_t>::N() const {
  return I_.size();
}

template<class scalar_t>
const SizeArray& Tensor<scalar_t>::size() const {
  return I_;
}

template<class scalar_t>
int Tensor<scalar_t>::size(const int n) const {
  if(n < 0 || n >= N()) {
    std::ostringstream oss;
    oss << "Tucker::Tensor::size(const int n): n = "
        << n << " is not in the range [0," << N() << ")";
    throw std::out_of_range(oss.str());
  }
  return I_[n];
}

/// \example Tucker_norm_test.cpp
template<class scalar_t>
scalar_t Tensor<scalar_t>::norm2() const
{
  // If this tensor has no entries, the norm is 0
  size_t numElements = getNumElements();
  if(numElements == 0)
    return 0;

  double normDouble =0;
  scalar_t norm = 0;
  size_t i;
  const scalar_t* dataPtr = data();
//#pragma omp parallel for reduction ( + : norm ) \
//  shared( numElements, dataPtr ) private(i)
  for(i=0; i<numElements; i++) {
    normDouble += (dataPtr[i]*dataPtr[i]);
  }
  norm = (scalar_t) normDouble;
  return norm;
}

template<class scalar_t>
scalar_t* Tensor<scalar_t>::data()
{
  if(data_)
    return data_;
  else
    throw std::runtime_error("Tucker::Tensor::data(): data was never allocated");
}

template<class scalar_t>
const scalar_t* Tensor<scalar_t>::data() const
{
  if(data_)
    return data_;
  else
    throw std::runtime_error("Tucker::Tensor::data(): data was never allocated");
}

template <class scalar_t>
bool isApproxEqual(const Tensor<scalar_t>* t1, const Tensor<scalar_t>* t2,
    scalar_t tol, bool verbose, bool ignoreSign)
{
  // If neither owns any data, they're not NOT equal...
  if(t1->getNumElements() == 0 && t2->getNumElements() == 0) {
    return true;
  }

  if(t1->size() != t2->size()) {
    std::cerr << "t1 and t2 are different sizes\n";
    return false;
  }

  scalar_t origNorm2 = t1->norm2();

  size_t numElements = t1->getNumElements();
  const scalar_t* t1Data = t1->data();
  const scalar_t* t2Data = t2->data();
  scalar_t errNorm2 = 0;
  for(size_t i=0; i<numElements; i++) {
    scalar_t err;
    if(ignoreSign) err = std::abs(std::abs(t1Data[i]) - std::abs(t2Data[i]));
    else err = std::abs(t1Data[i] - t2Data[i]);
    if(std::isnan(err)) {
      std::cerr << "Difference " << i << " is nan: "
          << t1Data[i] << " - " << t2Data[i] << " = "
          << t1Data[i] - t2Data[i] << std::endl;

      return false;
    }
    errNorm2 += (err*err);
  }
  std::cout << std::endl;

  
  scalar_t relErr = std::sqrt(errNorm2/origNorm2);
  if(verbose) {
    std::cout << "Relative error: " << relErr << std::endl;
  }

  if(relErr > tol){
    std::cout << "errNorm2:" << std::sqrt(errNorm2) << "  origNorm2: " << std::sqrt(origNorm2) << std::endl;
    std::cout << "relErr: " << relErr << std::endl;
    std::cout << "tol: " << tol << std::endl;
    return false;
  }
  return true;
}

template <class scalar_t>
void Tensor<scalar_t>::print(int precision) const
{
  // If this tensor doesn't have any entries, there's nothing to print
  size_t numElements = getNumElements();
  if(numElements == 0)
    return;

  const scalar_t* dataPtr = data();
  for(size_t i=0; i<numElements; i++) {
    std::cout << "data[" << i << "] = " << std::setprecision(precision) << dataPtr[i] << std::endl;
  }
}

template<class scalar_t>
size_t Tensor<scalar_t>::getNumElements() const
{
  return I_.prod();
}

template<class scalar_t>
void Tensor<scalar_t>::initialize()
{
  // If this tensor has no entries, it's already been initialized
  size_t numElements = getNumElements();
  if(numElements == 0)
    return;

  scalar_t* dataPtr = data();
  for(size_t i=0; i<numElements; i++) {
    dataPtr[i] = 0;
  }
}

template<class scalar_t>
void Tensor<scalar_t>::rand()
{
  size_t numElements = getNumElements();
  if(numElements == 0)
    return;

  scalar_t* dataPtr = data();
  for(size_t i=0; i<numElements; i++) {
    dataPtr[i] = std::rand();
  }
}

// Explicit instantiations to build static library for both single and double precision
template class Tensor<float>;
template bool isApproxEqual(const Tensor<float>*, const Tensor<float>*, float, bool, bool);

template class Tensor<double>;
template bool isApproxEqual(const Tensor<double>*, const Tensor<double>*, double, bool, bool);

} // end of namespace Tucker
