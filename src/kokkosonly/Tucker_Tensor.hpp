#ifndef TUCKER_KOKKOSONLY_TENSOR_HPP_
#define TUCKER_KOKKOSONLY_TENSOR_HPP_

#include "Tucker_SizeArray.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>

namespace TuckerKokkos{

namespace impl{
template<class Enable, class ScalarType, class ...Props>
struct TensorTraits;

template<class ScalarType> struct TensorTraits<void, ScalarType>{
  using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
  using view_type = Kokkos::View<ScalarType*, memory_space>;
};

template<class ScalarType, class MemSpace>
struct TensorTraits<
  std::enable_if_t< Kokkos::is_memory_space_v<MemSpace> >, ScalarType, MemSpace >
{
  using memory_space = MemSpace;
  using view_type = Kokkos::View<ScalarType*, memory_space>;
};
}//end namespace impl

template<class ScalarType, class ...Props>
class Tensor
{
  static_assert(std::is_floating_point_v<ScalarType>, "");
  using view_type = typename impl::TensorTraits<void, ScalarType, Props...>::view_type;

public:
  using traits = impl::TensorTraits<void, ScalarType, Props...>;

  Tensor() = default;
  Tensor(const SizeArray & szIn)
    : sizeArrayInfo_(szIn)
  {
    // Compute the total number of entries in this tensor
    const size_t numEntries = szIn.prod();
    data_ = view_type("tensorData", numEntries);
  }

  int rank() const{ return sizeArrayInfo_.size(); }

  const SizeArray& sizeArray() const{ return sizeArrayInfo_; }

  int extent(int mode) const { return sizeArrayInfo_[mode]; }

  size_t size() const{ return sizeArrayInfo_.prod(); }

  auto norm2Squared() const{
    const auto v = ::KokkosBlas::nrm2(data_);
    return v*v;
  }

  view_type data() const{ return data_; }

  void writeToStream(std::ostream & stream,
		     int precision = 2) const
  {
    auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data_);

    const size_t numElements = size();
    if(numElements == 0){ return; }

    for(size_t i=0; i<numElements; i++) {
      stream << "data[" << i << "] = "
	     << std::setprecision(precision)
	     << v_h(i) << std::endl;
    }
  }

  void fillRandom(ScalarType a, ScalarType b){
    Kokkos::Random_XorShift64_Pool<> pool(4543423);
    Kokkos::fill_random(data_, pool, a, b);
  }

private:
  view_type data_;
  SizeArray sizeArrayInfo_;
};

}
#endif
