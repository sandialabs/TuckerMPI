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

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "Tucker_SizeArray.hpp"

namespace Tucker {

/** \brief A sequential tensor
 *
 * Data is stored such that the unfolding \f$Y_0\f$ is column
 * major.  This means the flattening \f$Y_{N-1}\f$ is row-major,
 * and any other flattening \f$Y_n\f$ can be represented as a set
 * of \f$\prod\limits_{k=n+1}^{N-1}I_k\f$ row major matrices, each
 * of which is \f$I_n \times \prod\limits_{k=0}^{n-1}I_k\f$.
 *
 * Example: a \f$3 \times 4 \times 2\f$ tensor
 *
 * Mode 0 unfolding: \n
 * \f$\left[\begin{array}{cccccccc}
 * 0 & 3 & 6 & 9 & 12 & 15 & 18 & 21 \\
 * 1 & 4 & 7 & 10 & 13 & 16 & 19 & 22 \\
 * 2 & 5 & 8 & 11 & 14 & 17 & 20 & 23
 * \end{array}\right]\f$
 *
 * Mode 1 unfolding: \n
 * \f$\left[\begin{array}{ccc:ccc}
 * 0 & 1 & 2 & 12 & 13 & 14 \\
 * 3 & 4 & 5 & 15 & 16 & 17 \\
 * 6 & 7 & 8 & 18 & 19 & 20 \\
 * 9 & 10 & 11 & 21 & 22 & 23
 * \end{array}\right]\f$
 *
 * Mode 2 unfolding: \n
 * \f$\left[\begin{array}{cccccccccccc}
 * 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 \\
 * 12 & 13 & 14 & 15 & 16 & 17 & 18 & 19 & 20 & 21 & 22 & 23
 * \end{array}\right]\f$
 */
template<class scalar_t>
class Tensor {
public:
  /** \brief Constructor
   *
   * Creates a tensor with the dimensions specified by \a I
   * \param[in] I Specifies the dimensions of the tensor
   *
   * \note Copies the SizeArray data to a new SizeArray.  Allocates
   * memory for data_.
   *
   * \exception std::length_error Any entry of \a I is negative
   */
  Tensor(const SizeArray& I);

  /** \brief Destructor
   *
   * Deallocates memory for data_.
   */
  ~Tensor();

  //! Order/Number of dimensions
  int N() const;

  /** \brief Size of the tensor
   *
   * Returns a const reference to the SizeArray specifying the
   * dimensions of the tensor
   */
  const SizeArray& size() const;

  /** \brief Mode-\a n size
   *
   * Returns the size of the \a n-th mode.
   * \param[in] n The specified dimension
   *
   * \exception std::out_of_range \a n is not in the range [0,N)
   *
   * \note This is 0-based, not 1-based.
   */
  int size(const int n) const;

  /** \brief Compute the norm squared
   *
   * Returns the sum of every element squared
   * i.e. \f$e_0^2 + e_1^2 + e_2^2 + \cdots\f$
   *
   * \test Tucker_norm_test.cpp - Computes the norm of a
   * 4-dimensional tensor (\f$3 \times 5 \times 7 \times 11\f$)
   * randomly generated by MATLAB.  Compares the computed
   * norm to a gold standard solution, also generated by MATLAB.
   * If the computed norm is within \f$10^{-10}\f$ of the gold
   * standard, the test passes.
   */
  scalar_t norm2() const;

  /** \brief Data pointer
   *
   * We provide direct access to the data, i.e. a pointer to the
   * array that holds all the data.  This is needed for easy
   * compatibility with BLAS, LAPACK, and MPI
   *
   * \exception std::runtime_error data was never allocated
   */
  scalar_t* data();

  /** \brief Const data pointer
   *
   * We provide direct access to the data, i.e. a pointer to the
   * array that holds all the data.  This is needed for easy
   * compatibility with BLAS, LAPACK, and MPI
   *
   * \exception std::runtime_error data was never allocated
   */
  const scalar_t* data() const;

  /** \brief Prints the tensor to std::cout
   *
   * Tensor is printed as a single-dimensional array.
   *
   * Example: \n
   * <tt>
   * data[0] = 3.4\n
   * data[1] = 5.2\n
   * data[3] = 7.6
   * </tt>
   */
  void print(int precision = 2) const;

  /** \brief Returns the total number of elements in this tensor
   *
   * This number is computed on-the-fly and is not stored as part
   * of the class.
   */
  size_t getNumElements() const;

  /** \brief Initializes the entries of the tensor to 0
   *
   * This method is used by the MPI code when some processes do
   * not own any work.
   */
  void initialize();

  /** \brief Initializes the entries of the tensor with random values
   *
   * \warning This method does not seed the RNG.
   */
  void rand();
protected:
  /** \brief Matrix constructor
   *
   * This constructor is necessary so that Matrix can be a subclass
   * of Tensor.
   *
   * \note Allocates memory for data
   * \exception std::length_error \a nrows or \a ncols < 0
   */
  Tensor(int nrows, int ncols);

  Tensor(int nrows);

  //! Tensor size
  SizeArray I_;

  //! Tensor data
  scalar_t* data_;

private:
  /// @cond EXCLUDE
  // Copy constructor
  // We provide a private one with no implementation to disable it
  Tensor(const Tensor<scalar_t>& t);
  /// @endcond
};

/** \brief Determines whether two tensors are approximately the same
 *
 * This method first checks to see whether the two tensors are the
 * same dimension.  If not, returns false.  If they are the same size,
 * it checks every entry to see whether the entries are approximately
 * the same.  The maximum allowable difference between entries is
 * defined by the user.  This method is meant to be used for testing
 * the Tucker code, since bitwise comparison is a bad idea.
 *
 * \param t1 A tensor to compare
 * \param t2 A tensor to compare
 * \param tol The maximum allowable difference between entries
 */
template <class scalar_t>
bool isApproxEqual(const Tensor<scalar_t>* t1, const Tensor<scalar_t>* t2,
    scalar_t tol, bool verbose=false, bool ignoreSign = false);

} // end of namespace Tucker

#endif /* TENSOR_HPP_ */
