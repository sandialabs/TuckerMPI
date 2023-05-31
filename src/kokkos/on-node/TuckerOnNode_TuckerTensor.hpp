#ifndef TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{

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
  TuckerOnNode::Tensor<ScalarType, MemorySpace> G;
  std::vector< Kokkos::View<ScalarType*, MemorySpace> > eigenvalues;
  std::vector< Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> > singularvectors;
};


template <class ScalarType, class MemorySpace>
void print_eigenvalues(const TuckerTensor<ScalarType, MemorySpace> & factorization,
		       const std::string& filePrefix,
		       bool useLQ)
{
  const int nmodes = factorization.numDims();

  for(int mode=0; mode<nmodes; mode++) {
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";
    std::ofstream ofs(ss.str());
    // Determine the number of eigenvalues for this mode
    auto eigVals_view = factorization.eigValsAt(mode);
    auto eigVals_view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                    eigVals_view);
    const int nevals = eigVals_view.extent(0);

    // if(useLQ){
    //   for(int i=0; i<nevals; i++) {
    //     ofs << std::setprecision(16)
    //      << std::pow(factorization->singularValues[mode][i], 2)
    //      << std::endl;
    //   }
    // }
    // else{
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16)
      << eigVals_view_h(i)
      << std::endl;
      }
   //}
    ofs.close();
  }
}

}
#endif
