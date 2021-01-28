/*
 * Tucker_BlasWrapper.cpp
 *
 *  Created on: Jul 7, 2017
 *      Author: amklinv
 */

#include "Tucker_BlasWrapper.hpp"

namespace Tucker
{

void syrk(const char* uplo, const char* trans, const int* n,
    const int* k, const double* alpha, const double* a, const int* lda,
    const double* beta, double* c, const int* ldc)
{
  dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(const char* uplo, const char* trans, const int* n,
    const int* k, const float* alpha, const float* a, const int* lda,
    const float* beta, float* c, const int* ldc)
{
  ssyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syev(const char* jobz, const char* uplo, const int* n,
    double* a, const int* lda, double* w, double* work, int* lwork, int* info)
{
  dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}

void syev(const char* jobz, const char* uplo, const int* n,
    float* a, const int* lda, float* w, float* work, int* lwork, int* info)
{
  ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}

void swap(const int* n, double* dx, const int* incx,
    double* dy, const int* incy)
{
  dswap_(n, dx, incx, dy, incy);
}

void swap(const int* n, float* dx, const int* incx,
    float* dy, const int* incy)
{
  sswap_(n, dx, incx, dy, incy);
}

void copy(const int* n, const double* dx, const int* incx,
    double* dy, const int* incy)
{
  dcopy_(n, dx, incx, dy, incy);
}

void copy(const int* n, const float* dx, const int* incx,
    float* dy, const int* incy)
{
  scopy_(n, dx, incx, dy, incy);
}

void scal(const int* n, const double* da, double* dx, const int* incx)
{
  dscal_(n, da, dx, incx);
}

void scal(const int* n, const float* da, float* dx, const int* incx)
{
  sscal_(n, da, dx, incx);
}

void gemm(const char* transa, const char* transb, const int* m,
    const int* n, const int* k, const double* alpha, const double* a, const int* lda,
    const double* b, const int* ldb, const double* beta, double* c, const int* ldc)
{
  dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(const char* transa, const char* transb, const int* m,
    const int* n, const int* k, const float* alpha, const float* a, const int* lda,
    const float* b, const int* ldb, const float* beta, float* c, const int* ldc)
{
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

} // end namespace Tucker
