#ifndef IMPL_TUCKER_SOLVER_UTILS_HPP_
#define IMPL_TUCKER_SOLVER_UTILS_HPP_

#include <stdexcept>
#include <cstdio>

#include <Kokkos_Core.hpp>

#if defined KOKKOS_ENABLE_HIP
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#endif

#if defined KOKKOS_ENABLE_CUDA
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#if defined KOKKOS_ENABLE_CUDA
#define CUDA_CHECK(err)                                                  \
  do {                                                                   \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  } while (0)

#define CUBLAS_CHECK(err)                                                    \
  do {                                                                       \
    cublasStatus_t err_ = (err);                                             \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                     \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);   \
      throw std::runtime_error("cublas error");                              \
    }                                                                        \
  } while (0)

#define CUSOLVER_CHECK(err)                                                  \
  do {                                                                       \
    cusolverStatus_t err_ = (err);                                           \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                                   \
      std::printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cusolver error");                            \
    }                                                                        \
  } while (0)

namespace Tucker {
namespace impl {

class CublasHandle {
public:
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle(CublasHandle&&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  CublasHandle& operator=(CublasHandle&&) = delete;

  ~CublasHandle() { cublasDestroy(handle); }
  static cublasHandle_t get() {
    static CublasHandle h;
    return h.handle;
  }

private:
  cublasHandle_t handle;
  CublasHandle() { CUBLAS_CHECK( cublasCreate(&handle) ); }
};

class CusolverHandle {
public:
  CusolverHandle(const CusolverHandle&) = delete;
  CusolverHandle(CusolverHandle&&) = delete;
  CusolverHandle& operator=(const CusolverHandle&) = delete;
  CusolverHandle& operator=(CusolverHandle&&) = delete;

  ~CusolverHandle() { cusolverDnDestroy(handle); }
  static cusolverDnHandle_t get() {
    static CusolverHandle h;
    return h.handle;
  }

private:
  cusolverDnHandle_t handle;
  CusolverHandle() { CUSOLVER_CHECK( cusolverDnCreate(&handle) ); }
};

}
}

#endif

#if defined KOKKOS_ENABLE_HIP

#define ROCBLAS_CHECK(err)                                                  \
  do {                                                                      \
    rocblas_status err_ = (err);                                            \
    if (err_ != rocblas_status_success) {                                   \
      std::printf("rocblas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("rocblas error");                            \
    }                                                                       \
  } while (0)

namespace Tucker {
namespace impl {

class RocblasHandle {
public:
  RocblasHandle(const RocblasHandle&) = delete;
  RocblasHandle(RocblasHandle&&) = delete;
  RocblasHandle& operator=(const RocblasHandle&) = delete;
  RocblasHandle& operator=(RocblasHandle&&) = delete;

  ~RocblasHandle() { rocblas_destroy_handle(handle); }
  static rocblas_handle get() {
    static RocblasHandle h;
    return h.handle;
  }

private:
  rocblas_handle handle;
  RocblasHandle() { ROCBLAS_CHECK( rocblas_create_handle(&handle) ); }
};


}
}

#endif

#endif  // IMPL_TUCKER_SOLVER_UTILS_HPP_
