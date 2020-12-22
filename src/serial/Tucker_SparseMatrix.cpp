/*
 * SparseMatrix.cpp
 *
 *  Created on: Mar 14, 2017
 *      Author: amklinv
 */


#include "Tucker_SparseMatrix.hpp"
#include "Tucker_Util.hpp"
#include <cassert>

namespace Tucker {

template <class scalar_t>
SparseMatrix<scalar_t>::SparseMatrix(const int nrows, const int ncols, const int nnz)
{
  nrows_ = nrows;
  ncols_ = ncols;
  nnz_ = nnz;

  rows_ = Tucker::MemoryManager::safe_new_array<int>(nnz);
  cols_ = Tucker::MemoryManager::safe_new_array<int>(nnz);
  vals_ = Tucker::MemoryManager::safe_new_array<scalar_t>(nnz);
}

template <class scalar_t>
SparseMatrix<scalar_t>::~SparseMatrix() {
  Tucker::MemoryManager::safe_delete_array<int>(rows_,nnz_);
  Tucker::MemoryManager::safe_delete_array<int>(cols_,nnz_);
  Tucker::MemoryManager::safe_delete_array<scalar_t>(vals_,nnz_);
}

template <class scalar_t>
Matrix<scalar_t>* SparseMatrix<scalar_t>::multiply(const Matrix<scalar_t>* m, bool transp)
{
  assert(m != NULL);

  // Ensure that the dimensions are consistent
  if(transp) {
    assert(m->nrows() == nrows_);
  }
  else {
    assert(m->nrows() == ncols_);
  }

  // Allocate memory for result
  Matrix<scalar_t>* result;
  if(transp) {
    result = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(ncols_,m->ncols());
  }
  else {
    result = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(nrows_,m->ncols());
  }

  // Initialize result to 0
  result->initialize();

  scalar_t* resultData = result->data();
  const scalar_t* mData = m->data();
  for(int i=0; i<nnz_; i++) {
    for(int c=0; c<m->ncols(); c++) {
      if(transp) {
        // result(col(i),c) += val(i) m(row(i),c)
        resultData[cols_[i]+ncols_*c] += vals_[i]*mData[rows_[i]+m->nrows()*c];
      }
      else {
        // result(row(i),c) += val(i) m(col(i),c)
        resultData[rows_[i]+nrows_*c] += vals_[i]*mData[cols_[i]+m->nrows()*c];
      }
    } // end for(int c=0; c<m->ncols(); c++)
  } // end for(int i=0; i<nnz_; i++)

  return result;
}

template <class scalar_t>
Vector<scalar_t>* SparseMatrix<scalar_t>::multiply(const Vector<scalar_t>* v, bool transp)
{
  assert(v != NULL);

  // Ensure that the dimensions are consistent
  if(transp) {
    assert(v->nrows() == nrows_);
  }
  else {
    assert(v->nrows() == ncols_);
  }

  // Allocate memory for result
  Vector<scalar_t>* result;
  if(transp) {
    result = Tucker::MemoryManager::safe_new<Vector<scalar_t>>(ncols_);
  }
  else {
    result = Tucker::MemoryManager::safe_new<Vector<scalar_t>>(nrows_);
  }

  // Initialize result to 0
  result->initialize();

  scalar_t* resultData = result->data();
  const scalar_t* vData = v->data();
  for(int i=0; i<nnz_; i++) {
    if(transp) {
      // result(col(i)) += val(i) m(row(i))
      resultData[cols_[i]] += vals_[i]*vData[rows_[i]];
    }
    else {
      // result(row(i),c) += val(i) m(col(i),c)
      resultData[rows_[i]] += vals_[i]*vData[cols_[i]];
    }
  } // end for(int i=0; i<nnz_; i++)

  return result;
}

// Explicit instantiations to build static library for both single and double precision
template class SparseMatrix<float>;
template class SparseMatrix<double>;

} /* namespace Tucker */
