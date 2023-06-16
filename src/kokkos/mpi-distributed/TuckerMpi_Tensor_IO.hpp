#ifndef TUCKER_MPI_TENSOR_IO_HPP_
#define TUCKER_MPI_TENSOR_IO_HPP_

#include "Tucker_BoilerPlate_IO.hpp"
#include "TuckerMpi_MPIWrapper.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>

namespace TuckerMpi{

template <class ScalarType, class MemorySpace>
void import_tensor_binary(Tensor<ScalarType, MemorySpace> Y,
			  const char* filename)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(Y.getDistribution().ownNothing()) { return; }

  const int ndims = Y.getNumDimensions();
  int starts[ndims];
  int lsizes[ndims];
  int gsizes[ndims];
  for(int i=0; i<ndims; i++) {
    starts[i] = Y.getDistribution().getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y.getLocalSize(i);
    gsizes[i] = Y.getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray_<ScalarType>(ndims, gsizes, lsizes,
				      starts, MPI_ORDER_FORTRAN, &view);
  MPI_Type_commit(&view);

  // Open the file
  MPI_File fh;
  const MPI_Comm& comm = Y.getDistribution().getComm(true);
  int ret = MPI_File_open(comm, (char*)filename, MPI_MODE_RDONLY,
			  MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Set the view
  MPI_Offset disp = 0;
  MPI_File_set_view_<ScalarType>(fh, disp, view, "native", MPI_INFO_NULL);


  // Read the file
  size_t count = Y.getLocalNumEntries();
  assert(count <= std::numeric_limits<int>::max());
  if(rank == 0 && sizeof(ScalarType)*count > std::numeric_limits<int>::max()) {
    std::cout << "WARNING: We are attempting to call MPI_File_read_all to read ";
    Tucker::printBytes(sizeof(ScalarType)*count);
    std::cout << "Depending on your MPI implementation, this may fail "
              << "because you are trying to read over 2.1 GB.\nIf MPI_File_read_all"
              << " crashes, please try again with a more favorable processor grid.\n";
  }

  MPI_Status status;
  auto localTensorView = Y.getLocalTensor().data();
  ret = MPI_File_read_all_(fh, localTensorView.data(), (int)count, &status);
  int nread;
  MPI_Get_count_<ScalarType>(&status, &nread);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not read file " << filename << std::endl;
  }
  MPI_File_close(&fh);
  MPI_Type_free(&view);
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
    throw std::runtime_error("TuckerMpi::read_tensor_binary hardwired for one file for now");
  }
  import_tensor_binary(Y, temp.c_str());
}

} // end namespace Tucker
#endif
