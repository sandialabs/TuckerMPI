#ifndef TUCKER_KOKKOSONLY_TENSOR_IO_HPP_
#define TUCKER_KOKKOSONLY_TENSOR_IO_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_BoilerPlate_IO.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>

namespace TuckerOnNode{

template <class ScalarType, class MemorySpace>
void output_tensor_to_stream(Tensor<ScalarType, MemorySpace> X,
			     std::ostream & stream,
			     int precision = 2)
{
  auto data = X.data();
  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
  const size_t numElements = X.size();
  if(numElements == 0){ return; }

  for(size_t i=0; i<numElements; i++) {
    stream << "data[" << i << "] = "
	   << std::setprecision(precision)
	   << v_h(i) << std::endl;
  }
}

template <class ScalarType, class MemorySpace>
void import_tensor_binary(Tensor<ScalarType, MemorySpace> X,
			  const char* filename)
{
  auto view1d_d = X.data();
  auto view1d_h = Kokkos::create_mirror(view1d_d);
  Tucker::fill_rank1_view_from_binary_file(view1d_h, filename);
  Kokkos::deep_copy(view1d_d, view1d_h);
}

template <class ScalarType, class MemorySpace>
void read_tensor_binary(Tensor<ScalarType, MemorySpace> Y,
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

template <class ScalarType, class mem_space>
void export_tensor_binary(const Tensor<ScalarType, mem_space> & Y,
			  const char* filename)
{
  // const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  size_t numEntries = Y.size();
  // Open file
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::binary);
  assert(ofs.is_open());
  const ScalarType* data = Y.data().data();
  ofs.write((char*)data,numEntries*sizeof(ScalarType));
  ofs.close();
}

} // end namespace Tucker
#endif
