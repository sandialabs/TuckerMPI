#ifndef TUCKERMPI_TENSOR_IO_HPP_
#define TUCKERMPI_TENSOR_IO_HPP_

#include "./impl/TuckerMpi_MPIWrapper.hpp"
#include "./impl/TuckerMpi_prod_impl.hpp"
#include "TuckerMpi_Tensor.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include "Tucker_print_bytes.hpp"
#include "Tucker_boilerplate_view_io.hpp"
#include "Tucker_print_bytes.hpp"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>

namespace TuckerMpi{

template <class ScalarType, class ...Properties>
void read_tensor_binary(const int mpiRank,
      Tensor<ScalarType, Properties...> tensor,
      const std::string & filename)
{
  if(tensor.getDistribution().ownNothing()) { return; }

  const int ndims = tensor.rank();
  int starts[ndims];
  int lsizes[ndims];
  int gsizes[ndims];
  for(int i=0; i<ndims; i++) {
    starts[i] = tensor.getDistribution().getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = tensor.localExtent(i);
    gsizes[i] = tensor.globalExtent(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray_<ScalarType>(ndims, gsizes, lsizes,
              starts, MPI_ORDER_FORTRAN, &view);
  MPI_Type_commit(&view);

  // Open the file
  MPI_File fh;
  const MPI_Comm& comm = tensor.getDistribution().getComm(true);
  int ret = MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY,
        MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Set the view
  MPI_Offset disp = 0;
  MPI_File_set_view_<ScalarType>(fh, disp, view, "native", MPI_INFO_NULL);

  // Read the file
  size_t count = tensor.localSize();
  assert(count <= std::numeric_limits<size_t>::max());
  if(mpiRank == 0 && sizeof(ScalarType)*count > std::numeric_limits<size_t>::max()) {
    std::cout << "WARNING: We are attempting to call MPI_File_read_all to read ";
    Tucker::print_bytes_to_stream(std::cout, sizeof(ScalarType)*count);
    std::cout << "Depending on your MPI implementation, this may fail "
              << "because you are trying to read over 2.1 GB.\nIf MPI_File_read_all"
              << " crashes, please try again with a more favorable processor grid.\n";
  }

  MPI_Status status;
  auto localTensorView_d = tensor.localTensor().data();
  auto localTensorView_h = Kokkos::create_mirror(localTensorView_d);
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
void import_time_series(const int mpiRank,
      Tensor<ScalarType, Properties...> tensor,
      const std::vector<std::string> & filenames)
{
  using tensor_type = ::TuckerMpi::Tensor<ScalarType, Properties...>;
  using onnode_layout = typename tensor_type::traits::onnode_layout;
  static_assert(   std::is_same_v<onnode_layout, Kokkos::LayoutLeft>,
       "TuckerMpi::import_time_series: currently only supports tensors with LayoutLeft");

  const auto & distrib = tensor.getDistribution();
  if(distrib.ownNothing()) { return; }

  const int ndims = tensor.rank();
  std::vector<int> starts(ndims-1);
  std::vector<int> lsizes(ndims-1);
  std::vector<int> gsizes(ndims-1);
  for(int i=0; i<ndims-1; i++) {
    starts[i] = distrib.getMap(i, true)->getGlobalIndex(0);
    lsizes[i] = tensor.localExtent(i);
    gsizes[i] = tensor.globalExtent(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype mpiDataType;
  MPI_Type_create_subarray_<ScalarType>(ndims-1, gsizes.data(), lsizes.data(),
          starts.data(), MPI_ORDER_FORTRAN,
          &mpiDataType);
  MPI_Type_commit(&mpiDataType);

  const int nsteps = tensor.globalExtent(ndims-1);
  const auto stepMap = distrib.getMap(ndims-1,true);
  const auto & stepComm  = distrib.getProcessorGrid().getRowComm(ndims-1,true);
  auto localTensorView_d = tensor.localTensor().data();
  auto localTensorView_h = Kokkos::create_mirror(localTensorView_d);
  ScalarType * dataPtr   = localTensorView_h.data();

  auto sz = tensor.localDimensionsOnHost();
  const size_t count = impl::prod(sz, 0, ndims-2);
  assert(count <= std::numeric_limits<size_t>::max());
  if(mpiRank == 0 && 8*count > std::numeric_limits<size_t>::max()) {
    std::cout << "WARNING: We are attempting to call MPI_File_read_all to read ";
    ::Tucker::print_bytes_to_stream(std::cout, 8*count);
    std::cout << "Depending on your MPI implementation, this may fail "
              << "because you are trying to read over 2.1 GB.\nIf MPI_File_read_all"
              << " crashes, please try again with a more favorable processor grid.\n";
  }

  for(int step=0; step<nsteps; step++)
  {
    const std::string stepFilename = filenames[step];
    if(mpiRank == 0) {
      std::cout << "Reading file " << stepFilename << std::endl;
    }

    if (!stepMap->hasGlobalIndex(step)){
      std::cout << "write_tensor_binary_multifile: skipping for " << step << "\n";
      continue;
    }
    // int LO = stepMap->getLocalIndex(step);
    // if(LO < 0) { continue; }

    MPI_File fh;
    int ret = MPI_File_open(stepComm, (char*)stepFilename.c_str(),
          MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(ret != MPI_SUCCESS && mpiRank == 0) {
      std::cerr << "Error: Could not open file " << stepFilename << std::endl;
    }

    MPI_Offset disp = 0;
    MPI_File_set_view_<ScalarType>(fh, disp, mpiDataType, "native", MPI_INFO_NULL);
    MPI_Status status;
    ret = MPI_File_read_all_(fh, dataPtr, (int)count, &status);
    if(ret != MPI_SUCCESS && mpiRank == 0) {
      std::cerr << "Error: Could not read file " << stepFilename << std::endl;
      exit(1);
    }

    MPI_File_close(&fh);
    dataPtr += count;
  }
  MPI_Type_free(&mpiDataType);
  Kokkos::deep_copy(localTensorView_d, localTensorView_h);
}

template <class ScalarType, class ...Properties>
void read_tensor_binary(const int mpiRank,
      Tensor<ScalarType, Properties...> tensor,
      const std::vector<std::string> & filenames)
{
  const std::size_t fileCount = filenames.size();
  if (fileCount == 1){
    read_tensor_binary(mpiRank, tensor, filenames[0]);
  }
  else{

    const int tensorRank = tensor.rank();
    if(fileCount != (std::size_t)tensor.globalExtent(tensorRank-1)) {
      if(mpiRank == 0) {
  std::cerr << "ERROR: The number of filenames you provided is "
      << filenames.size() << ", but the extent of the tensor's last mode is "
      << tensor.globalExtent(tensorRank-1) << ".\nCalling MPI_Abort...\n";
      }
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    import_time_series(mpiRank, tensor, filenames);
  }
}

template <class ScalarType, class ...Properties>
void write_tensor_binary(const int mpiRank,
       Tensor<ScalarType, Properties...> tensor,
       const std::string & filename)
{

  using tensor_type = Tensor<ScalarType, Properties...>;
  using layout      = typename tensor_type::traits::onnode_layout;
  static_assert(std::is_same_v<layout, Kokkos::LayoutLeft>,
    "TuckerMpi::write_tensor_binary: only supports layoutLeft");

  if(tensor.getDistribution().ownNothing()) { return; }

  auto tensor_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), tensor);
  auto tensor_local_view_h = tensor_h.localTensor().data();

  const int ndims = tensor_h.rank();

  // Define data layout parameters
  std::vector<int> starts(ndims);
  std::vector<int> lsizes(ndims);
  std::vector<int> gsizes(ndims);
  for(int i=0; i<ndims; i++) {
    starts[i] = tensor_h.getDistribution().getMap(i, true)->getGlobalIndex(0);
    lsizes[i] = tensor_h.localExtent(i);
    gsizes[i] = tensor_h.globalExtent(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype mpiDt;
  MPI_Type_create_subarray_<ScalarType>(ndims, gsizes.data(), lsizes.data(), starts.data(),
          MPI_ORDER_FORTRAN, &mpiDt);
  MPI_Type_commit(&mpiDt);

  // Open the file
  MPI_File fh;
  const MPI_Comm& comm = tensor_h.getDistribution().getComm(true);
  int ret = MPI_File_open(comm, filename.c_str(),
        MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS && mpiRank == 0) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Set the view
  MPI_Offset disp = 0;
  MPI_File_set_view_<ScalarType>(fh, disp, mpiDt, "native", MPI_INFO_NULL);

  // Write the file
  size_t count = tensor_h.localSize();
  assert(count <= std::numeric_limits<size_t>::max());
  MPI_Status status;
  ret = MPI_File_write_all_(fh, tensor_local_view_h.data(), (int)count, &status);
  if(ret != MPI_SUCCESS && mpiRank == 0) {
    std::cerr << "Error: Could not write to file " << filename << std::endl;
  }
  MPI_File_close(&fh);
  MPI_Type_free(&mpiDt);

}

template <class ScalarType, class ...Properties>
void write_tensor_binary_multifile(const int mpiRank,
           Tensor<ScalarType, Properties...> tensor,
           const std::vector<std::string> & filenames)
{
  using tensor_type = ::TuckerMpi::Tensor<ScalarType, Properties...>;
  using onnode_layout = typename tensor_type::traits::onnode_layout;
  static_assert(   std::is_same_v<onnode_layout, Kokkos::LayoutLeft>,
       "TuckerMpi::import_time_series: currently only supports tensors with LayoutLeft");

  const auto & distrib = tensor.getDistribution();
  if(distrib.ownNothing()) { return; }

  const int ndims = tensor.rank();
  std::vector<int> starts(ndims-1);
  std::vector<int> lsizes(ndims-1);
  std::vector<int> gsizes(ndims-1);
  for(int i=0; i<ndims-1; i++) {
    starts[i] = distrib.getMap(i, true)->getGlobalIndex(0);
    lsizes[i] = tensor.localExtent(i);
    gsizes[i] = tensor.globalExtent(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype mpiDataType;
  MPI_Type_create_subarray_<ScalarType>(ndims-1, gsizes.data(), lsizes.data(),
          starts.data(), MPI_ORDER_FORTRAN,
          &mpiDataType);
  MPI_Type_commit(&mpiDataType);

  const int nsteps = tensor.globalExtent(ndims-1);
  const auto stepMap = distrib.getMap(ndims-1,true);
  const auto & stepComm  = distrib.getProcessorGrid().getRowComm(ndims-1,true);
  auto localTensorView_d = tensor.localTensor().data();
  auto localTensorView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localTensorView_d);
  ScalarType * dataPtr   = localTensorView_h.data();

  auto sz = tensor.localDimensionsOnHost();
  const size_t count = impl::prod(sz, 0, ndims-2);
  assert(count <= std::numeric_limits<size_t>::max());

  for(int step=0; step<nsteps; step++)
  {
    const std::string stepFilename = filenames[step];
    if(mpiRank == 0) {
      std::cout << "Writing file " << stepFilename << std::endl;
    }

    if (!stepMap->hasGlobalIndex(step)){
      std::cout << "write_tensor_binary_multifile: skipping for " << step << "\n";
      continue;
    }
    // int LO = stepMap->getLocalIndex(step);
    // if(LO < 0) { continue; }

    MPI_File fh;
    int ret = MPI_File_open(stepComm, (char*)stepFilename.c_str(),
          MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if(ret != MPI_SUCCESS && mpiRank == 0) {
      std::cerr << "Error: Could not open file " << stepFilename << std::endl;
    }

    MPI_Offset disp = 0;
    MPI_File_set_view_<ScalarType>(fh, disp, mpiDataType, "native", MPI_INFO_NULL);
    MPI_Status status;
    ret = MPI_File_write_all_(fh, dataPtr, (int)count, &status);
    if(ret != MPI_SUCCESS && mpiRank == 0) {
      std::cerr << "Error: Could not write to file " << stepFilename << std::endl;
    }

    MPI_File_close(&fh);
    dataPtr += count;
  }
  MPI_Type_free(&mpiDataType);
}

template <class ScalarType, class ...Properties>
void write_tensor_binary(const int mpiRank,
       Tensor<ScalarType, Properties...> tensor,
       const std::vector<std::string> & filenames)
{

  const int fileCount = filenames.size();
  if (fileCount == 1){
    write_tensor_binary(mpiRank, tensor, filenames[0]);
  }
  else{

    const int tensorRank = tensor.rank();
    if(fileCount != tensor.globalExtent(tensorRank-1)) {
      if(mpiRank == 0) {
  std::cerr << "ERROR: The number of filenames you provided is "
      << filenames.size() << ", but the extent of the tensor's last mode is "
      << tensor.globalExtent(tensorRank-1) << ".\nCalling MPI_Abort...\n";
      }
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    write_tensor_binary_multifile(mpiRank, tensor, filenames);
  }
}

} // end namespace Tucker
#endif  // TUCKERMPI_TENSOR_IO_HPP_
