
#ifndef TUCKER_IMPL_PRINT_BYTES_HPP_
#define TUCKER_IMPL_PRINT_BYTES_HPP_

#include <fstream>
#include <iomanip>

namespace Tucker{

template<class T>
void print_bytes_to_stream(std::ostream & out, T bytes)
{
  static_assert(std::is_integral_v<T>);

  const size_t KB = 1e3;
  const size_t MB = 1e6;
  const size_t GB = 1e9;
  const size_t TB = 1e12;

  if(bytes > TB) {
    out << std::setprecision(5) << bytes / (double)TB << " TB\n";
  }
  else if(bytes > GB) {
    out << std::setprecision(5) << bytes / (double)GB << " GB\n";
  }
  else if(bytes > MB) {
    out << std::setprecision(5) << bytes / (double)MB << " MB\n";
  }
  else if(bytes > KB) {
    out << std::setprecision(5) << bytes / (double)KB << " KB\n";
  }
  else {
    out << bytes << " bytes\n";
  }
}

}
#endif
