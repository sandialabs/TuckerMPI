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

// Scale an array
extern "C" void dscal_(const int*, const double*, double*, const int*);

extern "C" void sscal_(const int*, const float*, float*, const int*);

// Matrix matrix multiply
extern "C" void dgemm_(const char*, const char*, const int*,
    const int*, const int*, const double*, const double*, const int*,
    const double*, const int*, const double*, double*, const int*);

extern "C" void sgemm_(const char*, const char*, const int*,
    const int*, const int*, const float*, const float*, const int*,
    const float*, const int*, const float*, float*, const int*);

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

void scal(const int*, const double*, double*, const int*);

void scal(const int*, const float*, float*, const int*);

void gemm(const char*, const char*, const int*,
    const int*, const int*, const double*, const double*, const int*,
    const double*, const int*, const double*, double*, const int*);

void gemm(const char*, const char*, const int*,
    const int*, const int*, const float*, const float*, const int*,
    const float*, const int*, const float*, float*, const int*);

} // end namespace Tucker

#endif /* SERIAL_TUCKER_BLASWRAPPER_HPP_ */
