/*
 * Tucker_BlasWrapper.hpp
 *
 *  Created on: Jul 7, 2017
 *      Author: amklinv
 */

#ifndef SERIAL_TUCKER_BLASWRAPPER_HPP_
#define SERIAL_TUCKER_BLASWRAPPER_HPP_

namespace Tucker
{

/// @cond EXCLUDE

// Symmetric rank-k update
extern "C" void dsyrk_(const char*, const char*, const int*,
    const int*, const double*, const double*, const int*,
    const double*, double*, const int*);

extern "C" void ssyrk_(const char*, const char*, const int*,
    const int*, const float*, const float*, const int*,
    const float*, float*, const int*);

// Symmetric eigenvalue solver
extern "C" void dsyev_(const char*, const char*, const int*,
    double *, const int*, double*, double*, int*, int*);

extern "C" void ssyev_(const char*, const char*, const int*,
    float *, const int*, float*, float*, int*, int*);

// Swap two arrays
extern "C" void dswap_(const int*, double*, const int*,
    double*, const int*);

extern "C" void sswap_(const int*, float*, const int*,
    float*, const int*);

// Copy from one array to another
extern "C" void dcopy_(const int*, const double*, const int*,
    double*, const int*);

extern "C" void scopy_(const int*, const float*, const int*,
    float*, const int*);

// add two arrays
extern "C" void daxpy_(const int *n, const double *alpha, const double *x,
    const int *incx, double *y, const int *incy);

extern "C" void saxpy_(const int *n, const float *alpha, const float *x,
    const int *incx, float *y, const int *incy);

// Scale an array
extern "C" void dscal_(const int*, const double*, double*, const int*);

extern "C" void sscal_(const int*, const float*, float*, const int*);

// array norm
extern "C" double dnrm2_(const int *n, const double *x, const int *inc);

extern "C" float snrm2_(const int *n, const float *x, const int *inc);

// Matrix vector multiply
extern "C" void dgemv_(const char *transa, const int *m, const int *n,
    const double *alpha, const double *A, const int *lda, const double *x,
    const int *incx, const double *beta, double *y, const int *incy);

extern "C" void sgemv_(const char *transa, const int *m, const int *n,
    const float *alpha, const float *A, const int *lda, const float *x,
    const int *incx, const float *beta, float *y, const int *incy);

// Matrix matrix multiply
extern "C" void dgemm_(const char*, const char*, const int*,
    const int*, const int*, const double*, const double*, const int*,
    const double*, const int*, const double*, double*, const int*);

extern "C" void sgemm_(const char*, const char*, const int*,
    const int*, const int*, const float*, const float*, const int*,
    const float*, const int*, const float*, float*, const int*);

extern "C" void dgelq_(const int*, const int*, const double*, const int*, 
    const double*, const int*, const double*, const int*, const int*);

extern "C" void sgelq_(const int*, const int*, const float*, const int*, 
    const float*, const int*, const float*, const int*, const int*);

extern "C" void dgeqr_(const int*, const int*, const double*, const int*,
    const double*, const int*, const double*, const int*, const int*);

extern "C" void sgeqr_(const int*, const int*, const float*, const int*,
    const float*, const int*, const float*, const int*, const int*);

extern "C" void dtpqrt_(const int*, const int*, const int*, const int*,
    const double*, const int*, const double*, const int*, 
    const double*, const int*, const double*, const int*);

extern "C" void stpqrt_(const int*, const int*, const int*, const int*,
    const float*, const int*, const float*, const int*, 
    const float*, const int*, const float*, const int*);

extern "C" void dgesvd_(const char*, const char*, const int*, const int*,
    const double*, const int*, const double*, const double*, const int*, 
    const double*, const int*, const double*, const int*, const int*);

extern "C" void sgesvd_(const char*, const char*, const int*, const int*,
    const float*, const int*, const float*, const float*, const int*, 
    const float*, const int*, const float*, const int*, const int*);

extern "C" void dgeqrf_(const int*, const int*, const double*, const int*, 
    const double*, const double*, const int*, const int*);

extern "C" void sgeqrf_(const int*, const int*, const float*, const int*, 
    const float*, const float*, const int*, const int*);

extern "C" void dgeqrt_(const int*, const int*, const int*, const double*, 
    const int*, const double*, const int*, const double*, const int*);

extern "C" void sgeqrt_(const int*, const int*, const int* nb, const float*, 
    const int*, const float*, const int*, const float*, const int*);

extern "C" void dgelqf_(const int*, const int*, const double*, const int*, 
    const double*, const double*, const int*, const int*);

extern "C" void sgelqf_(const int*, const int*, const float*, const int*, 
    const float*, const float*, const int*, const int*);

extern "C" void dgelqt_(const int*, const int*, const int*, const double*, const int*,
    const double*, const int*, const double*, const int*);

extern "C" void sgelqt_(const int*, const int*, const int*, const float*, const int*,
    const float*, const int*, const float*, const int*);

/// @endcond

// Overloaded wrappers
void syrk(const char*, const char*, const int*,
    const int*, const double*, const double*, const int*,
    const double*, double*, const int*);

void syrk(const char*, const char*, const int*,
    const int*, const float*, const float*, const int*,
    const float*, float*, const int*);

void syev(const char*, const char*, const int*,
    double*, const int*, double*, double*, int*, int*);

void syev(const char*, const char*, const int*,
    float*, const int*, float*, float*, int*, int*);

void swap(const int*, double*, const int*,
    double*, const int*);

void swap(const int*, float*, const int*,
    float*, const int*);

void copy(const int*, const double*, const int*,
    double*, const int*);

void copy(const int*, const float*, const int*,
    float*, const int*);

void axpy(const int *n, const double *alpha, const double *x,
    const int *incx, double *y, const int *incy);

void axpy(const int *n, const float *alpha, const float *x,
    const int *incx, float *y, const int *incy);

void scal(const int*, const double*, double*, const int*);

void scal(const int*, const float*, float*, const int*);

double nrm2(const int *n, const double *x, const int *inc);

float nrm2(const int *n, const float *x, const int *inc);

void gemv(const char *trans, const int *m, const int *n, const double *alpha,
    const double *A, const int *lda, const double *x, const int *incx,
    const double *beta, double *y, const int *incy);

void gemv(const char *trans, const int *m, const int *n, const float *alpha,
    const float *A, const int *lda, const float *x, const int *incx,
    const float *beta, float *y, const int *incy);

void gemm(const char*, const char*, const int*,
    const int*, const int*, const double*, const double*, const int*,
    const double*, const int*, const double*, double*, const int*);

void gemm(const char*, const char*, const int*,
    const int*, const int*, const float*, const float*, const int*,
    const float*, const int*, const float*, float*, const int*);

void gelq(const int* m, const int* n, const double* a, const int* lda,
    const double* t, const int* tsize, const double* work, const int* lwork, const int* info);

void gelq(const int* m, const int* n, const float* a, const int* lda,
    const float* t, const int* tsize, const float* work, const int* lwork, const int* info);

void geqr(const int* m, const int* n, const double* a, const int* lda,
    const double* t, const int* tsize, const double* work, const int* lwork, const int* info);

void geqr(const int* m, const int* n, const float* a, const int* lda,
    const float* t, const int* tsize, const float* work, const int* lwork, const int* info);

void tpqrt(const int* m, const int* n, const int* l, const int* nb,
    const double* a, const int* lda, const double* b, const int* ldb, 
    const double* t, const int* ldt, const double* work, const int* info);

void tpqrt(const int* m, const int* n, const int* l, const int* nb,
    const float* a, const int* lda, const float* b, const int* ldb, 
    const float* t, const int* ldt, const float* work, const int* info);

void gesvd(const char* jobu, const char* jobvt, const int* m, const int* n,
    const double* a, const int* lda, const double* s, const double* u, const int* ldu, 
    const double* vt, const int* ldvt, const double* work, const int* lwork, const int* info);

void gesvd(const char* jobu, const char* jobvt, const int* m, const int* n,
    const float* a, const int* lda, const float* s, const float* u, const int* ldu, 
    const float* vt, const int* ldvt, const float* work, const int* lwork, const int* info);

void geqrf(const int* m, const int* n, const double* a, const int* lda, 
    const double* tau, const double* work, const int* lwork, const int* info);

void geqrf(const int* m, const int* n, const float* a, const int* lda, 
    const float* tau, const float* work, const int* lwork, const int* info);

void geqrt(const int* m, const int* n, const int* nb, const double* a, 
    const int* lda, const double* t, const int* ldt, const double* work, const int* info);

void geqrt(const int* m, const int* n, const int* nb, const float* a, 
    const int* lda, const float* t, const int* ldt, const float* work, const int* info);

void gelqf(const int* m, const int* n, const double* a, const int* lda, 
    const double* tau, const double* work, const int* lwork, const int* info);

void gelqf(const int* m, const int* n, const float* a, const int* lda, 
    const float* tau, const float* work, const int* lwork, const int* info);    

void gelqt(const int* m, const int* n, const int* mb, const double* a, const int* lda,
    const double* t, const int* ldt, const double* work, const int* info);

void gelqt(const int* m, const int* n, const int* mb, const float* a, const int* lda,
    const float* t, const int* ldt, const float* work, const int* info);

} // end namespace Tucker

#endif /* SERIAL_TUCKER_BLASWRAPPER_HPP_ */
