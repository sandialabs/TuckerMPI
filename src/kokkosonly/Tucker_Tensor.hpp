#ifndef TUCKER_KOKKOSONLY_TENSOR_HPP_
#define TUCKER_KOKKOSONLY_TENSOR_HPP_

#include "Tucker_SizeArray.hpp"
#include "Tucker_BoilerPlate_IO.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>

namespace TuckerKokkos{

namespace impl{
template<class Enable, class ScalarType, class ...Properties>
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


template<class ScalarType, class ...Properties>
class Tensor
{
  static_assert(std::is_floating_point_v<ScalarType>, "");
  using view_type = typename impl::TensorTraits<void, ScalarType, Properties...>::view_type;

public:
  using traits = impl::TensorTraits<void, ScalarType, Properties...>;

  Tensor() = default;
  Tensor(const SizeArray & szIn)
    : sizeArrayInfo_(szIn)
  {
    // Compute the total number of entries in this tensor
    const size_t numEntries = szIn.prod();
    data_ = view_type("tensorData", numEntries);
  }

  std::size_t rank() const{ return sizeArrayInfo_.size(); }

  const SizeArray& sizeArray() const{ return sizeArrayInfo_; }

  std::size_t extent(std::size_t mode) const { return sizeArrayInfo_[mode]; }

  size_t size() const{ return sizeArrayInfo_.prod(); }

  auto frobeniusNormSquared() const{
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

template <class ScalarType, class MemorySpace>
void import_tensor_binary(Tensor<ScalarType, MemorySpace> & X,
			const char* filename)
{
  auto view1d_d = X.data();
  auto view1d_h = Kokkos::create_mirror(view1d_d);
  fill_rank1_view_from_binary_file(view1d_h, filename);
  Kokkos::deep_copy(view1d_d, view1d_h);
}

template <class ScalarType, class MemorySpace>
void read_tensor_binary(Tensor<ScalarType, MemorySpace> & Y,
		      const char* filename)
{
  std::ifstream inStream(filename);
  std::string temp;
  int nfiles = 0;
  while(inStream >> temp) { nfiles++; }
  inStream.close();
  if(nfiles != 1) {
    throw std::runtime_error("readTensorBinary hardwired for one file only for now");
  }
  import_tensor_binary(Y, temp.c_str());
}

template <class scalar_t, class mem_space>
void export_tensor_binary(const Tensor<scalar_t, mem_space> & Y,
			const char* filename)
{
  // const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  size_t numEntries = Y.size();
  // Open file
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::binary);
  assert(ofs.is_open());
  const scalar_t* data = Y.data().data();
  ofs.write((char*)data,numEntries*sizeof(scalar_t));
  ofs.close();
}

}
#endif
