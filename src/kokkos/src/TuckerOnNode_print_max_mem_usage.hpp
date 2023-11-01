
#ifndef TUCKER_ON_NODE_PRINT_MAX_MEM_USAGE_HPP_
#define TUCKER_ON_NODE_PRINT_MAX_MEM_USAGE_HPP_

#include <sys/resource.h>
#include <iostream>

#include "Tucker_print_bytes.hpp"

namespace TuckerOnNode {

inline void print_max_mem_usage_to_stream(std::ostream & out)
{
  rusage usage;
  long my_maxrss;
  auto ret = getrusage(RUSAGE_SELF, &usage);
  std::size_t max_mem = usage.ru_maxrss * 1024; // ru_maxrss is in KB
  out << "Maximum local memory usage: ";
  Tucker::print_bytes_to_stream(out, max_mem);
}

}
#endif  // TUCKER_ON_NODE_PRINT_MAX_MEM_USAGE_HPP_
