#ifndef KOKKOSMPI_TUCKERMPI_TTM_HPP_
#define KOKKOSMPI_TUCKERMPI_TTM_HPP_

#include <cmath>
#include "TuckerMpi_Tensor.hpp"
#include "TuckerOnNode_ttm.hpp"
#include <unistd.h>

namespace TuckerMpi{

namespace impl{
template<class ViewT>
size_t prod(const ViewT & sz,
	    const int low, const int high,
	    const int defaultReturnVal = -1)
{
  if(low < 0 || high >= sz.extent(0)) {
    // std::cerr << "ERROR: prod(" << low << "," << high
    // 	      << ") is invalid because indices must be in the range [0,"
    // 	      << sz.extent(0) << ").  Returning " << defaultReturnVal << std::endl;
    return defaultReturnVal;
  }

  if(low > high) { return defaultReturnVal; }
  size_t result = 1;
  for(int j = low; j <= high; j++){ result *= sz[j]; }
  return result;
}
}//impl


template <class ScalarType, class ...TensorProperties>
void packForTTM(TuckerOnNode::Tensor<ScalarType, TensorProperties...> Y,
		int n,
		const Map* map)
{
  size_t nentries = Y.size();
  if(nentries == 0){ return; }

  int ndim = Y.rank();
  // If n is the last dimension, the data is already packed
  // (because the data is stored in row-major order)
  if(n == ndim-1) { return; }

  const int inc = 1;

  // Allocate memory
  size_t numEntries = Y.size();
  std::vector<ScalarType> tempMem(numEntries);

  const MPI_Comm& comm = map->getComm();
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  // Get the leading dimension of this tensor unfolding
  auto sa = Y.dimensions();
  size_t leadingDim = impl::prod(sa, 0,n-1,1);

  // Get the number of global rows of this tensor unfolding
  int nGlobalRows = map->getGlobalNumEntries();

  int RANK;
  MPI_Comm_rank(MPI_COMM_WORLD,&RANK);

  // Get pointer to tensor data
  ScalarType* tenData = Y.data().data();
  size_t stride = leadingDim*nGlobalRows;
  size_t tempMemOffset = 0;
  for(int rank=0; rank<nprocs; rank++)
  {
    int nLocalRows = map->getNumEntries(rank);
    size_t blockSize = leadingDim*nLocalRows;
    int rowOffset = map->getOffset(rank);
    for(size_t tensorOffset = rowOffset*leadingDim;
        tensorOffset < numEntries;
        tensorOffset += stride)
    {
      int tbs = (int)blockSize;
      Tucker::copy(&tbs, tenData+tensorOffset, &inc,
		   tempMem.data()+tempMemOffset, &inc);
      tempMemOffset += blockSize;
    }
  }

  // Copy data from temporary memory back to tensor
  int temp = (int)numEntries;
  Tucker::copy(&temp, tempMem.data(), &inc, tenData, &inc);
}

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
Distribution ttm_dist(Tensor<ScalarType, TensorProperties...> X,
	      int n,
	      Kokkos::View<ScalarType**, ViewProperties...> U,
	      bool Utransp,
	      std::size_t nnz_limit)
{
  const int nrows = Utransp ? U.extent(1) : U.extent(0);
  const int ndims = X.getNumDimensions();
  std::vector<int> newSize(ndims);
  for(int i=0; i<ndims; i++) {
    newSize[i] = (i == n) ? nrows : X.getGlobalSize(i);
  }

  Distribution dist(newSize, X.getDistribution().getProcessorGrid().getSizeArray());
  return dist;
}

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
auto ttm(Tensor<ScalarType, TensorProperties...> X,
	 int n,
	 Kokkos::View<ScalarType**, ViewProperties...> U,
	 bool Utransp,
	 std::size_t nnz_limit)
{
  int _myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &_myrank);

  using result_type = Tensor<ScalarType, TensorProperties...>;
  using local_tensor_type = typename result_type::traits::onnode_tensor_type;

  // Compute the number of rows for the resulting "matrix"
  const int nrows = Utransp ? U.extent(1) : U.extent(0);

  // Get the size of the new tensor
  const int ndims = X.getNumDimensions();
  std::vector<int> newSize(ndims);
  for(int i=0; i<ndims; i++) {
    newSize[i] = (i == n) ? nrows : X.getGlobalSize(i);
  }

  Distribution dist(newSize, X.getDistribution().getProcessorGrid().getSizeArray());
  result_type Y(dist);

  // Get the local part of the tensor
  auto localX = X.getLocalTensor();
  auto localY = Y.getLocalTensor();

  // Determine whether there are multiple MPI processes along this dimension
  int Pn = X.getDistribution().getProcessorGrid().getNumProcs(n,false);
  if(Pn == 1)
  {
    if (_myrank == 0){
      std::cout << "MPITTM: Pn==1 \n";
    }

    if(!X.getDistribution().ownNothing()) {
      TuckerOnNode::ttm(localX, n, U, localY, Utransp);
    }
  }
  else
  {
    // Pn != 1
    const MPI_Comm& comm = X.getDistribution().getProcessorGrid().getColComm(n,false);

    // Determine whether we must block the result
    // If the temporary storage is bigger than the tensor, we block instead
    int K = nrows;
    int Jn = Utransp ? U.extent(0) : U.extent(1);

    int uGlobalRows = Y.getGlobalSize(n);
    int uGlobalCols = X.getGlobalSize(n);
    const Map* xMap = X.getDistribution().getMap(n,false);
    const Map* yMap = Y.getDistribution().getMap(n,false);

    ScalarType* Uptr;
    assert(U.getNumElements() > 0);
    if(Utransp){
      Uptr = U.data() + xMap->getGlobalIndex(0);
    }
    else{
      Uptr = U.data() + xMap->getGlobalIndex(0)*uGlobalRows;
    }

    const int stride = Utransp ? uGlobalCols : uGlobalRows;

    // TTM either by a single reduce_scatter, or a series of reductions.
    // Reduce_scatter tends to be faster, so we try to use it if the
    // memory requirements are not prohibitive.
    // Compute the nnz of the largest tensor piece being stored by any process
    size_t max_lcl_nnz_x = 1;
    for(int i=0; i<ndims; i++) {
      max_lcl_nnz_x *= X.getDistribution().getMap(i,false)->getMaxNumEntries();
    }
    // Compute the nnz required for the reduce_scatter
    size_t nnz_reduce_scatter = 1;
    for(int i=0; i<ndims; i++) {
      if(i == n)
        nnz_reduce_scatter *= Y.getGlobalSize(n);
      else
        nnz_reduce_scatter *= X.getDistribution().getMap(i,false)->getMaxNumEntries();
    }

    // ********************************************************
    // If the required memory is small, we can do a single reduce_scatter
    // ********************************************************
#if 0
    if(nnz_reduce_scatter <= std::max(max_lcl_nnz_x,nnz_limit)){
#endif
      local_tensor_type localResult;
      if(X.getDistribution().ownNothing()) {
	std::vector<int> sz(ndims);
        for(int i=0; i<ndims; i++) {
          sz[i] = X.getLocalSize(i);
        }
        sz[n] = Y.getGlobalSize(n);
        localResult = local_tensor_type(sz);
      }
      else {

	// //sleep(_myrank*1);
	// std::cout << "_______BEFORE TTM_________" << _myrank << "\n";
	// Tucker::write_view_to_stream(std::cout, localX.data());
	// std::cout << uGlobalRows << " " << stride << "\n";
	// MPI_Barrier(MPI_COMM_WORLD);

        localResult = TuckerOnNode::ttm(localX, n, Uptr, uGlobalRows, stride, Utransp);
      }

      // sleep(5);
      // sleep(_myrank*1);
      // std::cout << "_______AFTER TTM_________" << _myrank << "\n";
      // Tucker::write_view_to_stream(std::cout, localResult.data());
      // MPI_Barrier(MPI_COMM_WORLD);
      // sleep(5);

      packForTTM(localResult, n, yMap);

      // sleep(_myrank*1);
      // std::cout << "_______PACK________" << _myrank << "\n";
      // Tucker::write_view_to_stream(std::cout, localResult.data(), 14);
      // std::cout << uGlobalRows << " " << stride << "\n";
      // MPI_Barrier(MPI_COMM_WORLD);
      // MPI_Barrier(MPI_COMM_WORLD);

      std::vector<ScalarType> sendBuf;
      if(localResult.size() > 0){
	sendBuf.resize(localResult.size());
	Tucker::copy_view_to_stdvec(localResult.data(), sendBuf);
      }
      ScalarType* recvBuf = nullptr;
      if(localY.size() > 0){
	recvBuf = localY.data().data();
      }

      int nprocs;
      MPI_Comm_size(comm,&nprocs);
      int recvCounts[nprocs];
      auto Ylsz = Y.getLocalSize();
      size_t multiplier = impl::prod(Ylsz,0,n-1,1) * impl::prod(Ylsz, n+1,ndims-1,1);

      for(int i=0; i<nprocs; i++) {
        size_t temp = multiplier*(yMap->getNumEntries(i));
        recvCounts[i] = (int)temp;
      }
      MPI_Reduce_scatter_(sendBuf.data(), recvBuf, recvCounts, MPI_SUM, comm);

#if 0
    }
    else
    {
      // ****************************
      // do a series of reductions
      // ****************************
      for(int root=0; root<Pn; root++) {
        int uLocalRows = yMap->getNumEntries(root);

        if(uLocalRows == 0) { continue; }

	local_tensor_type localResult;
        if(X.getDistribution().ownNothing()) {
	  std::vector<int> sz(ndims);
          for(int i=0; i<ndims; i++) {
            sz[i] = X.getLocalSize(i);
          }
          sz[n] = uLocalRows;
          localResult = local_tensor_type(sz);
        }
        else {
          localResult = Tucker::ttm(localX, n, Uptr, uLocalRows, stride, Utransp);
        }

        // Combine the local results with a reduce operation
        const ScalarType* sendBuf;
        if(localResult->getNumElements() > 0)
          sendBuf = localResult->data();
        else
          sendBuf = 0;
        ScalarType* recvBuf;
        if(localY.getNumElements() > 0)
          recvBuf = localY.data();
        else
          recvBuf = 0;
        size_t count = localResult->getNumElements();
        assert(count <= std::numeric_limits<int>::max());

        if(count > 0) {
          MPI_Reduce_(sendBuf, recvBuf, (int)count, MPI_SUM, root, comm);
        }

        // Increment the data pointer
        if(Utransp){ Uptr += (uLocalRows*stride); }
	else{ Uptr += uLocalRows; }
      } // end for i = 0 .. Pn-1
    } // end if K >= Jn/Pn
#endif
  } // end if Pn != 1

  return Y;
}

} // end namespace TuckerMpi

#endif /* MPI_TUCKERMPI_TTM_HPP_ */
