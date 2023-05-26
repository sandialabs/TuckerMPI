
#ifndef TUCKER_BOILERPLATE_IO_UTIL_HPP_
#define TUCKER_BOILERPLATE_IO_UTIL_HPP_

#include <Kokkos_Core.hpp>
#include <fstream>

namespace TuckerKokkos{

template <class DataType, class ...Properties>
void fill_rank1_view_from_binary_file(Kokkos::View<DataType, Properties...> & v,
				 const char* filename)
{
  using view_type = Kokkos::View<DataType, Properties...>;
  using mem_space = typename view_type::memory_space;
  using value_type = typename view_type::value_type;

  static_assert(view_type::rank == 1);
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, mem_space>::accessible);

  // Get the maximum file size we can read
  // const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  std::ifstream ifs;
  ifs.open(filename, std::ios::in | std::ios::binary);
  assert(ifs.is_open());

  std::streampos begin, end, size;
  begin = ifs.tellg();
  ifs.seekg(0, std::ios::end);
  end = ifs.tellg();
  size = end - begin;
  //std::cout << "Reading " << size << " bytes...\n";
  assert(size == v.extent(0)*sizeof(value_type));

  // Read the file
  // auto view1d_d = X.data();
  // auto view1d_h = Kokkos::create_mirror(view1d_d);
  value_type* data = v.data();
  ifs.seekg(0, std::ios::beg);
  ifs.read((char*)data, size);
  ifs.close();
}


}// end namespace TuckerKokkos

#endif /* TUCKER_IO_UTIL_HPP_ */
