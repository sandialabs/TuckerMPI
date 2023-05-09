#ifndef TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_

#include <Kokkos_Core.hpp>

namespace TuckerKokkos{

template<class ScalarType, class MemorySpace>
class TuckerTensor
{
public:
  TuckerTensor(const int ndims) : N(ndims){
    assert(ndims > 0);
  }

  auto getFactorMatrix(int n){
    return Kokkos::subview(U, n, Kokkos::ALL, Kokkos::ALL);
  }

  void pushBack(Kokkos::View<ScalarType*, MemorySpace> ein){
    eigenvalues.emplace_back(ein);
  }

  void pushBack(Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> ein){
    singularvectors.emplace_back(ein);
  }

  auto eigValsAt(int i) const{ return eigenvalues[i]; }
  auto eigVecsAt(int i) const{ return singularvectors[i]; }
  int numDims() const{ return N; }
  auto const & getG() const{ return G; }
  auto & getG(){ return G; }

private:
  int N;
  Tensor<ScalarType, MemorySpace> G;
  Kokkos::View<ScalarType***, Kokkos::LayoutLeft, MemorySpace> U;
  std::vector< Kokkos::View<ScalarType*, MemorySpace> > eigenvalues;
  std::vector< Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> > singularvectors;
};

}
#endif
