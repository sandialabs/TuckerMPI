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

void gelq(const int* m, const int* n, const double* a, const int* lda,
    const double* t, const int* tsize, const double* work, const int* lwork, const int* info)
{
  dgelq_(m, n, a, lda, t, tsize, work, lwork, info);
}

void gelq(const int* m, const int* n, const float* a, const int* lda,
    const float* t, const int* tsize, const float* work, const int* lwork, const int* info)
{
  sgelq_(m, n, a, lda, t, tsize, work, lwork, info);
}

void geqr(const int* m, const int* n, const double* a, const int* lda,
    const double* t, const int* tsize, const double* work, const int* lwork, const int* info)
{
  dgeqr_(m, n, a, lda, t, tsize, work, lwork, info);
}

void geqr(const int* m, const int* n, const float* a, const int* lda,
    const float* t, const int* tsize, const float* work, const int* lwork, const int* info)
{
  sgeqr_(m, n, a, lda, t, tsize, work, lwork, info);
}

void tpqrt(const int* m, const int* n, const int* l, const int* nb,
    const double* a, const int* lda, const double* b, const int* ldb,
    const double* t, const int* ldt, const double* work, const int* info)
{
  dtpqrt_(m, n, l, nb, a, lda, b, ldb, t, ldt, work, info);
}

void tpqrt(const int* m, const int* n, const int* l, const int* nb,
    const float* a, const int* lda, const float* b, const int* ldb,
    const float* t, const int* ldt, const float* work, const int* info)
{
  stpqrt_(m, n, l, nb, a, lda, b, ldb, t, ldt, work, info);
}

void gesvd(const char* jobu, const char* jobvt, const int* m, const int* n,
    const double* a, const int* lda, const double* s, const double* u, const int* ldu,
    const double* vt, const int* ldvt, const double* work, const int* lwork, const int* info)
{
  dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

void gesvd(const char* jobu, const char* jobvt, const int* m, const int* n,
    const float* a, const int* lda, const float* s, const float* u, const int* ldu,
    const float* vt, const int* ldvt, const float* work, const int* lwork, const int* info)
{
  sgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

void geqrf(const int* m, const int* n, const double* a, const int* lda,
    const double* tau, const double* work, const int* lwork, const int* info)
{
  dgeqrf_(m, n, a, lda, tau, work, lwork, info);
}

void geqrf(const int* m, const int* n, const float* a, const int* lda,
    const float* tau, const float* work, const int* lwork, const int* info)
{
  sgeqrf_(m, n, a, lda, tau, work, lwork, info);
}

void geqrt(const int* m, const int* n, const int* nb, const double* a,
    const int* lda, const double* t, const int* ldt, const double* work, const int* info)
{
  dgeqrt_(m, n, nb, a, lda, t, ldt, work, info);
}

void geqrt(const int* m, const int* n, const int* nb, const float* a,
    const int* lda, const float* t, const int* ldt, const float* work, const int* info)
{
  sgeqrt_(m, n, nb, a, lda, t, ldt, work, info);
}

void gelqf(const int* m, const int* n, const double* a, const int* lda,
    const double* tau, const double* work, const int* lwork, const int* info)
{
  dgelqf_(m, n, a, lda, tau, work, lwork, info);
}

void gelqf(const int* m, const int* n, const float* a, const int* lda,
    const float* tau, const float* work, const int* lwork, const int* info)
{
  sgelqf_(m, n, a, lda, tau, work, lwork, info);
}

void gelqt(const int* m, const int* n, const int* mb, const double* a, const int* lda,
    const double* t, const int* ldt, const double* work, const int* info)
{
  dgelqt_(m, n, mb, a, lda, t, ldt, work, info);
}

void gelqt(const int* m, const int* n, const int* mb, const float* a, const int* lda,
    const float* t, const int* ldt, const float* work, const int* info)
{
  sgelqt_(m, n, mb, a, lda, t, ldt, work, info);
}

} // end namespace Tucker
