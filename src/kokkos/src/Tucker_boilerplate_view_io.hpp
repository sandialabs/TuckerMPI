
#ifndef TUCKER_BOILERPLATE_IO_UTIL_HPP_
#define TUCKER_BOILERPLATE_IO_UTIL_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <fstream>
#include <vector>
#include <iomanip>

namespace Tucker{

template <class DataType, class ...Properties>
void fill_rank1_view_from_binary_file(const Kokkos::View<DataType, Properties...> & v,
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

template <class DataType, class ...Properties>
void write_view_to_stream(std::ostream & out,
			  const Kokkos::View<DataType, Properties...> & v,
			  int precision = 8)
{
  using view_type = Kokkos::View<DataType, Properties...>;
  static_assert(view_type::rank <= 2, "view must have rank <= 2");
  static_assert(std::is_same_v<typename view_type::array_layout, Kokkos::LayoutRight> ||
		std::is_same_v<typename view_type::array_layout, Kokkos::LayoutLeft>,
		"view must have LayoutLeft or Right");

  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  if constexpr(view_type::rank == 1){
    for (std::size_t i=0; i<v_h.extent(0); ++i){
      out << std::setprecision(precision) << v_h(i) << "\n";
    }
  }
  else if constexpr(view_type::rank == 2){
    for (std::size_t i=0; i<v_h.extent(0); ++i){
      for (std::size_t j=0; j<v_h.extent(1); ++j){
	out << std::setprecision(precision) << v_h(i,j) << "  ";
      }
      out << " \n ";
    }
  }
}

template <class DataType, class ...Properties>
void write_view_to_stream_inline(std::ostream & out,
				 const Kokkos::View<DataType, Properties...> & v,
				 int precision = 8)
{
  using view_type = Kokkos::View<DataType, Properties...>;
  static_assert(view_type::rank == 1, "view must have rank == 1");
  static_assert(std::is_same_v<typename view_type::array_layout, Kokkos::LayoutRight> ||
		std::is_same_v<typename view_type::array_layout, Kokkos::LayoutLeft>,
		"view must have LayoutLeft or Right");

  auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  for (std::size_t i=0; i<v_h.extent(0); ++i){
    out << std::setprecision(precision) << v_h(i) << " ";
  }
}

template <class DataType, class ...Properties>
void export_view_binary(const Kokkos::View<DataType, Properties...> & v,
			const char* filename)
{
  using view_type = Kokkos::View<DataType, Properties...>;
  using value_type = typename view_type::non_const_value_type;

  const size_t numEntries = v.size();
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::binary);
  assert(ofs.is_open());
  if (v.span_is_contiguous()){
    auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    ofs.write((char*)v_h.data(), numEntries*sizeof(value_type));
    ofs.close();
  }
  else{
    throw std::runtime_error("export_view_binary: currently only supports contiguous Views");
  }
}

}// end namespace Tucker
#endif
