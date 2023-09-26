#ifndef TUCKERONNODE_TENSOR_IO_HPP_
#define TUCKERONNODE_TENSOR_IO_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include "Tucker_boilerplate_view_io.hpp"

namespace TuckerOnNode{

template <class ScalarType, class ...Properties>
void read_tensor_binary(Tensor<ScalarType, Properties...> tensor,
			const std::string & filename)
{
  std::cout << " filename = " << filename << '\n';
  auto tensor_h = Tucker::create_mirror(tensor);
  Tucker::fill_rank1_view_from_binary_file(tensor_h.data(), filename);
  Tucker::deep_copy(tensor, tensor_h);
}

template <class ScalarType, class ...Properties>
void read_tensor_binary(Tensor<ScalarType, Properties...> Y,
			const std::vector<std::string> & filenames)
{
  if(filenames.size() != 1) {
    throw std::runtime_error("TuckerOnNode::read_tensor_binary: only supports one file for now");
  }
  read_tensor_binary(Y, filenames[0]);
}

template <class ScalarType, class mem_space>
void write_tensor_binary(const Tensor<ScalarType, mem_space> & tensor,
			 const std::string & filename)
{

  using tensor_type = Tensor<ScalarType, mem_space>;
  using layout      = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<layout, Kokkos::LayoutLeft> ||
		std::is_same_v<layout, Kokkos::LayoutRight>,
		"export_tensor_binary: only supports layoutLeft or Right");

  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tensor.data());
  // const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  size_t numEntries = tensor.size();
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::binary);
  assert(ofs.is_open());
  const ScalarType* data = v_h.data();
  ofs.write((char*)data,numEntries*sizeof(ScalarType));
  ofs.close();
}

template <class ScalarType, class ...Properties>
void write_tensor_binary(Tensor<ScalarType, Properties...> tensor,
			 const std::vector<std::string> & filenames)
{
  if(filenames.size() != 1) {
    throw std::runtime_error("TuckerMpi::write_tensor_binary: only supports one file for now");
  }
  write_tensor_binary(tensor, filenames[0]);
}

} // end namespace Tucker
#endif  // TUCKERONNODE_TENSOR_IO_HPP_
