#ifndef TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_

#include <Kokkos_Core.hpp>

namespace TuckerKokkos{

template<class ScalarType, class MemorySpace>
class TuckerTensor
{
public:
  TuckerTensor(const int ndims)
    : N(ndims), eigenvalues(ndims), singularvectors(ndims)
  {
    assert(ndims > 0);
  }

  Kokkos::View<ScalarType*, MemorySpace> eigValsAt(int n) const {
    return eigenvalues[n];
  }

  Kokkos::View<ScalarType*, MemorySpace> & eigValsAt(int n) {
    return eigenvalues[n];
  }

  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> eigVecsAt(int n){
    return singularvectors[n];
  }

  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> eigVecsAt(int n) const{
    return singularvectors[n];
  }

  int numDims() const{ return N; }
  auto const & getG() const{ return G; }
  auto & getG(){ return G; }

private:
  int N;
  Tensor<ScalarType, MemorySpace> G;
  std::vector< Kokkos::View<ScalarType*, MemorySpace> > eigenvalues;
  std::vector< Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> > singularvectors;
};

}
#endif
