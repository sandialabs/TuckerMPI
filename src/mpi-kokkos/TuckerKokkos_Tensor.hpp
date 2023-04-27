#ifndef TENSOR_KOKKOS_HPP_
#define TENSOR_KOKKOS_HPP_

// #include <impl/Kokkos_MemorySpace.hpp>
// #include "Tucker_SizeArray.hpp"

namespace TuckerKokkos {

template<class scalar_t, class MemorySpace>
class Tensor
{
public:
    using data_type = Kokkos::View<scalar_t*, MemorySpace>;

    // Constructor
    Tensor(SizeArray sa): I_(sa) { /* init view */ };

    // Size of the tensor
    const SizeArray& size() const { return I_; }

    // Kokkos::View data
    data_type data() const { return data_; }

private:
    // Tensor size
    SizeArray I_;

    // Tensor data
    data_type data_;
};

} /* namespace TuckerMPI */

#endif /* TENSOR_MPI_HPP_ */