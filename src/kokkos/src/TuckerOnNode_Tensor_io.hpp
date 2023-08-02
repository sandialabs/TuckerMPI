#ifndef TUCKER_KOKKOSONLY_TENSOR_IO_HPP_
#define TUCKER_KOKKOSONLY_TENSOR_IO_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include "Tucker_boilerplate_view_io.hpp"

namespace TuckerOnNode{

template <class ScalarType, class ...Properties>
void output_tensor_to_stream(Tensor<ScalarType, Properties...> X,
			     std::ostream & stream,
			     int precision = 2)
{
  auto X_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), X);
  auto v_h = X_h.data();
  const size_t numElements = X_h.size();
  if(numElements == 0){ return; }

  for(size_t i=0; i<numElements; i++) {
    stream << "data[" << i << "] = "
	   << std::setprecision(precision)
	   << v_h(i) << std::endl;
  }
}

template <class ScalarType, class ...Properties>
void import_tensor_binary(Tensor<ScalarType, Properties...> X,
			  const char* filename)
{
  auto X_h = Tucker::create_mirror(X);
  Tucker::fill_rank1_view_from_binary_file(X_h.data(), filename);
  Tucker::deep_copy(X, X_h);
}

template <class ScalarType, class ...Properties>
void read_tensor_binary(Tensor<ScalarType, Properties...> Y,
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
  using tensor_type = Tensor<ScalarType, mem_space>;
  using layout      = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<layout, Kokkos::LayoutLeft> ||
		std::is_same_v<layout, Kokkos::LayoutRight>,
		"export_tensor_binary: only supports layoutLeft or Right");

  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y.data());
  // const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  size_t numEntries = Y.size();
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::binary);
  assert(ofs.is_open());
  const ScalarType* data = v_h.data();
  ofs.write((char*)data,numEntries*sizeof(ScalarType));
  ofs.close();
}

template <class ScalarType, class MemorySpace>
void write_tensor_binary(const Tensor<ScalarType, MemorySpace> X,
			  const char* filename)
{
  // Count the number of filenames
  std::ifstream inStream(filename);

  std::string temp;
  int nfiles = 0;
  while(inStream >> temp) {
    nfiles++;
  }

  inStream.close();

  if(nfiles == 1) {
    export_tensor_binary(X,temp.c_str());
  }
  else {
    throw std::runtime_error("write_tensor_binary using multiple files missing impl");

    // int ndims = X.rank();
    // if(nfiles != X.extent(ndims-1)) {
    //   std::ostringstream oss;
    //   oss << "TuckerOnNode::writeTensorBinary: "
    //       << "The number of filenames you provided is "
    //       << nfiles << ", but the dimension of the tensor's last mode is "
    //       << X.extent(ndims-1);
    //   throw std::runtime_error(oss.str());
    // }
    // // exportTimeSeries(Y,filename); >> TOREMOVE?
  }
}

} // end namespace Tucker
#endif
