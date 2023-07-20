#ifndef TUCKER_KOKKOS_MPI_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOS_MPI_TUCKERTENSOR_HPP_

#include "TuckerMpi_Tensor.hpp"
#include "./impl/Tucker_TuckerTensor_impl.hpp"

namespace TuckerMpi{

template <class ...Args>
using TuckerTensor = Tucker::impl::TuckerTensor<false, Args...>;

}
#endif
