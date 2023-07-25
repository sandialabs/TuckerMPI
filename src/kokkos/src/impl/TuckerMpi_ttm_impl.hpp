#ifndef TUCKER_KOKKOS_MPI_TTM_IMPL_HPP_
#define TUCKER_KOKKOS_MPI_TTM_IMPL_HPP_

#include "TuckerMpi_Tensor.hpp"
#include "TuckerOnNode_ttm.hpp"
#include <cmath>
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

template <class ScalarType, class ...TensorProperties>
void packForTTM(TuckerOnNode::Tensor<ScalarType, TensorProperties...> Y,
		int n,
		const Map* map)
{
  using tensor_type = TuckerOnNode::Tensor<ScalarType, TensorProperties...>;
  using mem_space = typename tensor_type::traits::memory_space;
  namespace KE = Kokkos::Experimental;

  if(Y.size() == 0){ return; }

  // If n is the last dimension, the data is already packed
  // (because the data is stored in row-major order)
  if(n == Y.rank()-1) { return; }

  // Create empty view
  size_t numEntries = Y.size();
  Kokkos::View<ScalarType*, mem_space> tempMem("tempMem", numEntries);

  const MPI_Comm& comm = map->getComm();
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  // Get the leading dimension of this tensor unfolding
  auto sa = Y.dimensionsOnHost();
  size_t leadingDim = impl::prod(sa, 0,n-1,1);

  // Get the number of global rows of this tensor unfolding
  int nGlobalRows = map->getGlobalNumEntries();

  auto y_data_view = Y.data();

  size_t stride = leadingDim*nGlobalRows;
  size_t tempMemOffset = 0;
  const int inc = 1;
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

      auto it_first_from = KE::begin(y_data_view)+tensorOffset;
      auto it_first_to = KE::begin(tempMem)+tempMemOffset;
      Kokkos::parallel_for(tbs, KOKKOS_LAMBDA (const int i){
        const int shift = inc*i;
        *(it_first_to + shift) = *(it_first_from + shift);
      });

      tempMemOffset += blockSize;
    }
  }

  // Copy data from temporary memory back to tensor
  KE::copy(typename mem_space::execution_space(), tempMem, y_data_view);
}

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
Distribution ttm_dist(Tensor<ScalarType, TensorProperties...> X,
	      int n,
	      Kokkos::View<ScalarType**, ViewProperties...> U,
	      bool Utransp,
	      std::size_t nnz_limit)
{
  const int nrows = Utransp ? U.extent(1) : U.extent(0);
  const int ndims = X.rank();
  std::vector<int> newSize(ndims);
  for(int i=0; i<ndims; i++) {
    newSize[i] = (i == n) ? nrows : X.globalExtent(i);
  }

  Distribution dist(newSize, X.getDistribution().getProcessorGrid().getSizeArray());
  return dist;
}

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
auto ttm_impl(Tensor<ScalarType, TensorProperties...> X,
	      int n,
	      Kokkos::View<ScalarType**, ViewProperties...> U,
	      bool Utransp,
	      std::size_t nnz_limit)
{
  int _myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &_myrank);

  using result_type = Tensor<ScalarType, TensorProperties...>;
  using local_tensor_type = typename result_type::traits::onnode_tensor_type;

  using U_view = Kokkos::View<ScalarType**, ViewProperties...>;
  using U_mem_space = typename U_view::memory_space;

  // Compute the number of rows for the resulting "matrix"
  const int nrows = Utransp ? U.extent(1) : U.extent(0);

  // Get the size of the new tensor
  const int ndims = X.rank();
  std::vector<int> newSize(ndims);
  for(int i=0; i<ndims; i++) {
    newSize[i] = (i == n) ? nrows : X.globalExtent(i);
  }

  Distribution dist(newSize, X.getDistribution().getProcessorGrid().getSizeArray());
  result_type Y(dist);
  auto localX = X.localTensor();
  auto localY = Y.localTensor();

  // Determine whether there are multiple MPI processes along this dimension
  int Pn = X.getDistribution().getProcessorGrid().getNumProcs(n,false);
  if(Pn == 1)
  {
    if (_myrank == 0){ std::cout << "MPITTM: Pn==1 \n"; }
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
    //int K = nrows;
    int Jn = Utransp ? U.extent(0) : U.extent(1);

    int uGlobalRows = Y.globalExtent(n);
    int uGlobalCols = X.globalExtent(n);
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
        nnz_reduce_scatter *= Y.globalExtent(n);
      else
        nnz_reduce_scatter *= X.getDistribution().getMap(i,false)->getMaxNumEntries();
    }

    // ********************************************************
    // If the required memory is small, we can do a single reduce_scatter
    // ********************************************************
    // if(nnz_reduce_scatter <= std::max(max_lcl_nnz_x,nnz_limit))
    // {
      local_tensor_type localResult;
      if(X.getDistribution().ownNothing()) {
	std::vector<int> sz(ndims);
        for(int i=0; i<ndims; i++) {
          sz[i] = X.localExtent(i);
        }
        sz[n] = Y.globalExtent(n);
        localResult = local_tensor_type(sz);
      }
      else
      {
	std::vector<int> I(localX.rank());
	for(int i=0; i<I.size(); i++) {
	  I[i] = (i != n) ? localX.extent(i) : uGlobalRows;
	}
	localResult = local_tensor_type(I);
	Kokkos::LayoutStride layout(localX.extent(n), 1, localResult.extent(n), stride);
	using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutStride, U_mem_space,
					 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
	umv_type Aum(Uptr, layout);
	TuckerOnNode::ttm(localX, n, Aum, localResult, Utransp);
      }

      packForTTM(localResult, n, yMap);

      std::vector<ScalarType> sendBuf;
      if(localResult.size() > 0){
	sendBuf.resize(localResult.size());
	Tucker::impl::copy_view_to_stdvec(localResult.data(), sendBuf);
      }

      auto localYview_h = Kokkos::create_mirror_view(localY.data());
      ScalarType* recvBuf = nullptr;
      if(localY.size() > 0){
	recvBuf = localYview_h.data(); //localY.data().data();
      }

      int nprocs;
      MPI_Comm_size(comm,&nprocs);
      int recvCounts[nprocs];
      auto Ylsz = Y.localDimensionsOnHost();
      size_t multiplier = impl::prod(Ylsz,0,n-1,1) * impl::prod(Ylsz, n+1,ndims-1,1);

      for(int i=0; i<nprocs; i++) {
        size_t temp = multiplier*(yMap->getNumEntries(i));
        recvCounts[i] = (int)temp;
      }
      MPI_Reduce_scatter_(sendBuf.data(), recvBuf, recvCounts, MPI_SUM, comm);
      Kokkos::deep_copy(localY.data(), localYview_h);

    // }
    // else{
    //   throw std::runtime_error("TuckerMpi: ttm: impl using series of reduction missing");
    // }
  } // end if Pn != 1

  return Y;
}

}} // end namespace TuckerMpi::impl

#endif /* MPI_TUCKERMPI_TTM_HPP_ */
