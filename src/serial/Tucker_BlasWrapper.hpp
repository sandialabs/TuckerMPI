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

// Symmetric eigenvalue solver
extern "C" void dsyev_(const char*, const char*, const int*,
    double *, const int*, double*, double*, int*, int*);

// Swap two arrays
extern "C" void dswap_(const int*, double*, const int*,
    double*, const int*);

// Copy from one array to another
extern "C" void dcopy_(const int*, const double*, const int*,
    double*, const int*);

// Scale an array
extern "C" void dscal_(const int*, const double*, double*, const int*);

// Matrix matrix multiply
extern "C" void dgemm_(const char*, const char*, const int*,
    const int*, const int*, const double*, const double*, const int*,
    const double*, const int*, const double*, double*, const int*);

/// @endcond

template<typename T>
void syrk(const char*, const char*, const int*,
    const int*, const T*, const T*, const int*,
    const T*, T*, const int*);

template<typename T>
void syev(const char*, const char*, const int*,
    T*, const int*, T*, T*, int*, int*);

template<typename T>
void swap(const int*, T*, const int*,
    T*, const int*);

template<typename T>
void copy(const int*, const T*, const int*,
    T*, const int*);

template<typename T>
void scal(const int*, const T*, T*, const int*);

template<typename T>
void gemm(const char*, const char*, const int*,
    const int*, const int*, const T*, const T*, const int*,
    const T*, const int*, const T*, T*, const int*);

} // end namespace Tucker

#endif /* SERIAL_TUCKER_BLASWRAPPER_HPP_ */
