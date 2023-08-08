#ifndef TUCKER_MPI_TENSOR_IO_HPP_
#define TUCKER_MPI_TENSOR_IO_HPP_

#include "./impl/TuckerMpi_MPIWrapper.hpp"
#include "TuckerMpi_Tensor.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include "Tucker_print_bytes.hpp"
#include "Tucker_boilerplate_view_io.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>

namespace TuckerMpi{

template <class ScalarType, class ...Properties>
void read_tensor_binary(Tensor<ScalarType, Properties...> Y,
			const std::string & filename)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(Y.getDistribution().ownNothing()) { return; }

  const int ndims = Y.rank();
  int starts[ndims];
  int lsizes[ndims];
  int gsizes[ndims];
  for(int i=0; i<ndims; i++) {
    starts[i] = Y.getDistribution().getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y.localExtent(i);
    gsizes[i] = Y.globalExtent(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray_<ScalarType>(ndims, gsizes, lsizes,
				      starts, MPI_ORDER_FORTRAN, &view);
  MPI_Type_commit(&view);

  // Open the file
  MPI_File fh;
  const MPI_Comm& comm = Y.getDistribution().getComm(true);
  int ret = MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY,
			  MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Set the view
  MPI_Offset disp = 0;
  MPI_File_set_view_<ScalarType>(fh, disp, view, "native", MPI_INFO_NULL);

  // Read the file
  size_t count = Y.localSize();
  assert(count <= std::numeric_limits<int>::max());
  if(rank == 0 && sizeof(ScalarType)*count > std::numeric_limits<int>::max()) {
    std::cout << "WARNING: We are attempting to call MPI_File_read_all to read ";
    Tucker::print_bytes_to_stream(std::cout, sizeof(ScalarType)*count);
    std::cout << "Depending on your MPI implementation, this may fail "
              << "because you are trying to read over 2.1 GB.\nIf MPI_File_read_all"
              << " crashes, please try again with a more favorable processor grid.\n";
  }

  MPI_Status status;
  auto localTensorView_d = Y.localTensor().data();
  auto localTensorView_h = Kokkos::create_mirror(localTensorView_d);
  //auto localTensorView = Y.localTensor().data();
  ret = MPI_File_read_all_(fh, localTensorView_h.data(), (int)count, &status);
  int nread;
  MPI_Get_count_<ScalarType>(&status, &nread);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not read file " << filename << std::endl;
  }
  MPI_File_close(&fh);
  Kokkos::deep_copy(localTensorView_d, localTensorView_h);
  MPI_Type_free(&view);
}

template <class ScalarType, class ...Properties>
void read_tensor_binary(Tensor<ScalarType, Properties...> Y,
			const std::vector<std::string> & filenames)
{
  if(filenames.size() != 1) {
    throw std::runtime_error("TuckerMpi::read_tensor_binary: only supports one file for now");
  }
  read_tensor_binary(Y, filenames[0]);
}

template <class ScalarType, class ...Properties>
void write_tensor_binary(const int mpiRank,
			 Tensor<ScalarType, Properties...> Y,
			 const std::string & filename)
{

  using tensor_type = Tensor<ScalarType, Properties...>;
  using layout      = typename tensor_type::traits::onnode_layout;
  static_assert(std::is_same_v<layout, Kokkos::LayoutLeft>,
		"TuckerMpi::write_tensor_binary: only supports layoutLeft");

  if(Y.getDistribution().ownNothing()) { return; }

  auto Y_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), Y);
  auto Y_local_view_h = Y_h.localTensor().data();

  const int ndims = Y_h.rank();

  // Define data layout parameters
  std::vector<int> starts(ndims);
  std::vector<int> lsizes(ndims);
  std::vector<int> gsizes(ndims);
  for(int i=0; i<ndims; i++) {
    starts[i] = Y_h.getDistribution().getMap(i, true)->getGlobalIndex(0);
    lsizes[i] = Y_h.localExtent(i);
    gsizes[i] = Y_h.globalExtent(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype mpiDt;
  MPI_Type_create_subarray_<ScalarType>(ndims, gsizes.data(), lsizes.data(), starts.data(),
					MPI_ORDER_FORTRAN, &mpiDt);
  MPI_Type_commit(&mpiDt);

  // Open the file
  MPI_File fh;
  const MPI_Comm& comm = Y_h.getDistribution().getComm(true);
  int ret = MPI_File_open(comm, filename.c_str(),
			  MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS && mpiRank == 0) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Set the view
  MPI_Offset disp = 0;
  MPI_File_set_view_<ScalarType>(fh, disp, mpiDt, "native", MPI_INFO_NULL);

  // Write the file
  size_t count = Y_h.localSize();
  assert(count <= std::numeric_limits<int>::max());
  MPI_Status status;
  ret = MPI_File_write_all_(fh, Y_local_view_h.data(), (int)count, &status);
  if(ret != MPI_SUCCESS && mpiRank == 0) {
    std::cerr << "Error: Could not write to file " << filename << std::endl;
  }
  MPI_File_close(&fh);
  MPI_Type_free(&mpiDt);

}

template <class ScalarType, class ...Properties>
void write_tensor_binary(const int mpiRank,
			 Tensor<ScalarType, Properties...> Y,
			 const std::vector<std::string> & filenames)
{
  if(filenames.size() != 1) {
    throw std::runtime_error("TuckerMpi::write_tensor_binary: only supports one file for now");
  }
  write_tensor_binary(mpiRank, Y, filenames[0]);
}

} // end namespace Tucker
#endif
