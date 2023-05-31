
#ifndef TUCKER_BOILERPLATE_IO_UTIL_HPP_
#define TUCKER_BOILERPLATE_IO_UTIL_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <fstream>
#include <vector>

namespace Tucker{

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
  assert(size == v.extent(0)*sizeof(value_type));

  // we need to read into std::vector and then copy to view
  // because this has to work for possibly non-contiguous views
  std::vector<value_type> stdVec(v.extent(0));
  ifs.seekg(0, std::ios::beg);
  ifs.read((char*)stdVec.data(), size);
  ifs.close();

  std::copy(stdVec.cbegin(), stdVec.cend(), Kokkos::Experimental::begin(v));
}


}// end namespace Tucker

#endif /* TUCKER_IO_UTIL_HPP_ */
