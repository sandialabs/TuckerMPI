#ifndef TUCKER_KOKKOSONLY_TENSOR_HPP_
#define TUCKER_KOKKOSONLY_TENSOR_HPP_

#include "Tucker_SizeArray.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>

namespace TuckerKokkos{

template<class ScalarType, class MemorySpace>
class Tensor
{
  static_assert(std::is_floating_point_v<ScalarType>, "");

  using view_type = Kokkos::View<ScalarType*, MemorySpace>;
  using exespace = typename view_type::execution_space;

public:
  Tensor() = default;
  Tensor(const SizeArray & I) : I_(I.size())
  {
    // Copy the SizeArray
    for(int i=0; i<I.size(); i++) {
      if(I[i] < 0) {
	std::ostringstream oss;
	oss << "TuckerKokkos::Tensor(const SizeArray& I): I["
	    << i << "] = " << I[i] << " < 0.";
	throw std::length_error(oss.str());
      }
      I_[i] = I[i];
    }

    // Compute the total number of entries in this tensor
    const size_t numEntries = getNumElements();
    data_ = view_type("tensorData", numEntries);
  }

  int N() const{
    return I_.size();
  }

  const SizeArray& size() const{
    return I_;
  }

  int size(const int n) const{
    if(n < 0 || n >= N()) {
      std::ostringstream oss;
      oss << "Tucker::Tensor::size(const int n): n = "
	  << n << " is not in the range [0," << N() << ")";
      throw std::out_of_range(oss.str());
    }
    return I_[n];
  }

  size_t getNumElements() const{
    return I_.prod();
  }

  ScalarType norm2() const{
    const auto v = ::KokkosBlas::nrm2(data_);
    return v*v;
  }

  const view_type data() const{
    return data_;
  }

  void print(int precision = 2) const{
    auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data_);

    // If this tensor doesn't have any entries, there's nothing to print
    size_t numElements = getNumElements();
    if(numElements == 0){
      return;
    }
    const ScalarType* dataPtr = v_h.data();
    for(size_t i=0; i<numElements; i++) {
      std::cout << "data[" << i << "] = "
		<< std::setprecision(precision) << dataPtr[i] << std::endl;
    }
  }

  void initialize(){
    throw std::runtime_error("Tensor::initialize missing impl");
  }

  void rand(ScalarType a, ScalarType b){
    Kokkos::Random_XorShift64_Pool<exespace> pool(4543423);
    Kokkos::fill_random(data_, pool, a, b);
  }

private:
  view_type data_;
  SizeArray I_;
};

}
#endif
