
#ifndef TUCKER_MPI_PRINT_MAX_MEM_USAGE_HPP_
#define TUCKER_MPI_PRINT_MAX_MEM_USAGE_HPP_

#include <sys/resource.h>
#include <iostream>
#include "mpi.h"

#include "Tucker_print_bytes.hpp"

namespace TuckerMpi {

inline void print_max_mem_usage_to_stream(const MPI_Comm& comm, std::ostream & out)
{
  rusage usage;
  auto ret = getrusage(RUSAGE_SELF, &usage);
  std::size_t my_max_mem = usage.ru_maxrss * 1024; // ru_maxrss is in KB

  int mpi_rank = 0;
  size_t max_mem = 0;
  size_t tot_mem = 0;
  MPI_Datatype mpi_type = sizeof(size_t) == 8 ? MPI_UNSIGNED_LONG : MPI_UNSIGNED;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Reduce(&my_max_mem, &max_mem, 1, mpi_type, MPI_MAX, 0, comm);
  MPI_Reduce(&my_max_mem, &tot_mem, 1, mpi_type, MPI_SUM, 0, comm);

  if (mpi_rank == 0) {
    out << "Maximum local memory usage: ";
    Tucker::print_bytes_to_stream(out, max_mem);
    out << "Maximum global memory usage: ";
    Tucker::print_bytes_to_stream(out, tot_mem);
  }
}

}
#endif  // TUCKER_MPI_PRINT_MAX_MEM_USAGE_HPP_
