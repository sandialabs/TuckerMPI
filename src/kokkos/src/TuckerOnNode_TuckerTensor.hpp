#ifndef TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "./impl/Tucker_TuckerTensor_impl.hpp"

namespace TuckerOnNode{

template <class ...Args>
using TuckerTensor = Tucker::impl::TuckerTensor<true, Args...>;

}
#endif
