#ifndef IMPL_TUCKER_SOLVER_UTILS_HPP_
#define IMPL_TUCKER_SOLVER_UTILS_HPP_

#include <stdexcept>
#include <cstdio>

#include <Kokkos_Core.hpp>

#if defined KOKKOS_ENABLE_HIP
#include <rocsolver/rocsolver.h>
#endif

#if defined KOKKOS_ENABLE_CUDA
#include <cusolverDn.h>
#endif

#if defined KOKKOS_ENABLE_CUDA
#define CUDA_CHECK(err)                                                 \
  do {                                                                  \
    cudaError_t err_ = (err);                                           \
    if (err_ != cudaSuccess) {                                          \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                           \
    }                                                                   \
  } while (0)

#define CUSOLVER_CHECK(err)                                             \
  do {                                                                  \
    cusolverStatus_t err_ = (err);                                      \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                              \
      std::printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cusolver error");                       \
    }                                                                   \
  } while (0)

namespace Tucker {
namespace impl {

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

#endif  // IMPL_TUCKER_SOLVER_UTILS_HPP_
