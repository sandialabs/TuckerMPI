#ifndef IMPL_TUCKERMPI_TTM_IMPL_HPP_
#define IMPL_TUCKERMPI_TTM_IMPL_HPP_

#include "TuckerMpi_prod_impl.hpp"
#include "TuckerOnNode_ttm.hpp"
#include <cmath>
#include <unistd.h>

namespace TuckerMpi{
namespace impl{

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
void ttm_impl_use_single_reduce_scatter(Tensor<ScalarType, TensorProperties...> X,
					Tensor<ScalarType, TensorProperties...> & Y,
					int n,
					Kokkos::View<ScalarType**, ViewProperties...> U,
					bool Utransp)
{
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
  if (mpiRank == 0){ std::cout << "MPITTM: single reduce_scatter \n"; }
#endif

  auto localX = X.localTensor();
  auto localY = Y.localTensor();
  const int ndims = X.rank();

  using result_type = Tensor<ScalarType, TensorProperties...>;
  using local_tensor_type = typename result_type::traits::onnode_tensor_type;

  using U_view = Kokkos::View<ScalarType**, ViewProperties...>;
  using U_mem_space = typename U_view::memory_space;
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutStride, U_mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const MPI_Comm& comm = X.getDistribution().getProcessorGrid().getColComm(n,false);

  int uGlobalRows = Y.globalExtent(n);
  int uGlobalCols = X.globalExtent(n);
  const Map* xMap = X.getDistribution().getMap(n,false);
  const Map* yMap = Y.getDistribution().getMap(n,false);

  ScalarType* Uptr;
  assert(U.size() > 0);
  if(Utransp){ Uptr = U.data() + xMap->getGlobalIndex(0); }
  else{ Uptr = U.data() + xMap->getGlobalIndex(0)*uGlobalRows; }

  const int stride = Utransp ? uGlobalCols : uGlobalRows;

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
      for(int i=0; i<(int)I.size(); i++) {
	I[i] = (i != n) ? localX.extent(i) : uGlobalRows;
      }
      localResult = local_tensor_type(I);

      const std::size_t Unrows = (Utransp) ? localX.extent(n) : localResult.extent(n);
      const std::size_t Uncols = (Utransp) ? localResult.extent(n) : localX.extent(n);
      Kokkos::LayoutStride layout(Unrows, 1, Uncols, stride);
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
    recvBuf = localYview_h.data();
  }

  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  int recvCounts[nprocs];
  auto Ylsz = Y.localDimensionsOnHost();
  size_t multiplier = impl::prod(Ylsz,0,n-1,1) * impl::prod(Ylsz, n+1,ndims-1,1);

  for(int i=0; i<nprocs; i++) {
    size_t temp = multiplier*(yMap->getNumEntries(i));
    recvCounts[i] = (int)temp;
  }
  MPI_Reduce_scatter_(sendBuf.data(), recvBuf, recvCounts, MPI_SUM, comm);
  Kokkos::deep_copy(localY.data(), localYview_h);
}


template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
void ttm_impl_use_series_of_reductions(Tensor<ScalarType, TensorProperties...> X,
				       Tensor<ScalarType, TensorProperties...> & Y,
				       int n,
				       Kokkos::View<ScalarType**, ViewProperties...> U,
				       bool Utransp)
{
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
  if (mpiRank == 0){ std::cout << "MPITTM: series of reductions \n"; }
#endif

  auto localX = X.localTensor();
  auto localY = Y.localTensor();
  const int ndims = X.rank();

  using result_type = Tensor<ScalarType, TensorProperties...>;
  using local_tensor_type = typename result_type::traits::onnode_tensor_type;

  using U_view = Kokkos::View<ScalarType**, ViewProperties...>;
  using U_mem_space = typename U_view::memory_space;
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutStride, U_mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const MPI_Comm& comm = X.getDistribution().getProcessorGrid().getColComm(n,false);

  int uGlobalRows = Y.globalExtent(n);
  int uGlobalCols = X.globalExtent(n);
  const Map* xMap = X.getDistribution().getMap(n,false);
  const Map* yMap = Y.getDistribution().getMap(n,false);

  ScalarType* Uptr;
  assert(U.size() > 0);
  if(Utransp){ Uptr = U.data() + xMap->getGlobalIndex(0); }
  else{ Uptr = U.data() + xMap->getGlobalIndex(0)*uGlobalRows; }

  const int stride = Utransp ? uGlobalCols : uGlobalRows;

  auto localYview_h = Kokkos::create_mirror_view(localY.data());
  const int Pn = X.getDistribution().getProcessorGrid().getNumProcs(n, false);
  for(int root=0; root<Pn; root++)
  {
    int uLocalRows = yMap->getNumEntries(root);
    if(uLocalRows == 0) { continue; }

    // Compute the local TTM
    local_tensor_type localResult;
    if(X.getDistribution().ownNothing()) {
      std::vector<int> sz(ndims);
      for(int i=0; i<ndims; i++) {
	sz[i] = X.localExtent(i);
      }
      sz[n] = uLocalRows;
      localResult = local_tensor_type(sz);
    }
    else
      {
	std::vector<int> I(localX.rank());
	for(int i=0; i<(int)I.size(); i++) {
	  I[i] = (i != n) ? localX.extent(i) : uLocalRows;
	}
	localResult = local_tensor_type(I);

	const std::size_t Unrows = (Utransp) ? localX.extent(n) : localResult.extent(n);
	const std::size_t Uncols = (Utransp) ? localResult.extent(n) : localX.extent(n);
	Kokkos::LayoutStride layout(Unrows, 1, Uncols, stride);
	umv_type Aum(Uptr, layout);
	TuckerOnNode::ttm(localX, n, Aum, localResult, Utransp);
      }

    // Combine the local results with a reduce operation
    std::vector<ScalarType> sendBuf;
    if(localResult.size() > 0){
      sendBuf.resize(localResult.size());
      Tucker::impl::copy_view_to_stdvec(localResult.data(), sendBuf);
    }

    ScalarType* recvBuf = nullptr;
    if(localY.size() > 0){
      recvBuf = localYview_h.data();
    }
    size_t count = localResult.size();
    assert(count <= std::numeric_limits<std::size_t>::max());

    if(count > 0) {
      MPI_Reduce_(sendBuf.data(), recvBuf, (int)count, MPI_SUM, root, comm);
    }

    if(Utransp){ Uptr += (uLocalRows*stride);}
    else{ Uptr += uLocalRows; }
  }
}



#if 0
template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
auto ttm_impl(Tensor<ScalarType, TensorProperties...> X,
	      int n,
	      Kokkos::View<ScalarType**, ViewProperties...> U,
	      bool Utransp,
	      std::size_t nnz_limit)
{
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  // using result_type = Tensor<ScalarType, TensorProperties...>;
  // using local_tensor_type = typename result_type::traits::onnode_tensor_type;

  using U_view = Kokkos::View<ScalarType**, ViewProperties...>;
  using U_mem_space = typename U_view::memory_space;
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutStride, U_mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // // Compute the number of rows for the resulting "matrix"
  // const int nrows = Utransp ? U.extent(1) : U.extent(0);
  // // Get the size of the new tensor
  // const int ndims = X.rank();
  // std::vector<int> newSize(ndims);
  // for(int i=0; i<ndims; i++) {
  //   newSize[i] = (i == n) ? nrows : X.globalExtent(i);
  // }

  // Distribution dist(newSize, X.getDistribution().getProcessorGrid().getSizeArray());
  // result_type Y(dist);
  auto localX = X.localTensor();
  auto localY = Y.localTensor();

  // Determine whether there are multiple MPI processes along this dimension
  int Pn = X.getDistribution().getProcessorGrid().getNumProcs(n,false);
  if(Pn == 1)
  {
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank == 0){ std::cout << "MPITTM: Pn==1 \n"; }
#endif

    if(!X.getDistribution().ownNothing()) {
      TuckerOnNode::ttm(localX, n, U, localY, Utransp);
    }
  }
  else
  {
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank == 0){ std::cout << "MPITTM: Pn != 1 \n"; }
#endif

    // Pn != 1
    const MPI_Comm& comm = X.getDistribution().getProcessorGrid().getColComm(n,false);

    int uGlobalRows = Y.globalExtent(n);
    int uGlobalCols = X.globalExtent(n);
    const Map* xMap = X.getDistribution().getMap(n,false);
    const Map* yMap = Y.getDistribution().getMap(n,false);

    ScalarType* Uptr;
    assert(U.size() > 0);
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

    // *******************************************************************
    // If the required memory is small, we can do a single reduce_scatter
    // *******************************************************************
    if(nnz_reduce_scatter <= std::max(max_lcl_nnz_x, nnz_limit))
    {
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
      if (mpiRank == 0){ std::cout << "MPITTM: single reduce_scatter \n"; }
#endif

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
      	for(int i=0; i<(int)I.size(); i++) {
      	  I[i] = (i != n) ? localX.extent(i) : uGlobalRows;
      	}
      	localResult = local_tensor_type(I);

	const std::size_t Unrows = (Utransp) ? localX.extent(n) : localResult.extent(n);
	const std::size_t Uncols = (Utransp) ? localResult.extent(n) : localX.extent(n);
	Kokkos::LayoutStride layout(Unrows, 1, Uncols, stride);
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
	      recvBuf = localYview_h.data();
      }

      int nprocs;
      MPI_Comm_size(comm, &nprocs);
      int recvCounts[nprocs];
      auto Ylsz = Y.localDimensionsOnHost();
      size_t multiplier = impl::prod(Ylsz,0,n-1,1) * impl::prod(Ylsz, n+1,ndims-1,1);

      for(int i=0; i<nprocs; i++) {
        size_t temp = multiplier*(yMap->getNumEntries(i));
        recvCounts[i] = (int)temp;
      }
      MPI_Reduce_scatter_(sendBuf.data(), recvBuf, recvCounts, MPI_SUM, comm);
      Kokkos::deep_copy(localY.data(), localYview_h);

    }
    else
    {
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
      if (mpiRank == 0){ std::cout << "MPITTM: use series of reductions \n"; }
#endif

      auto localYview_h = Kokkos::create_mirror_view(localY.data());
      for(int root=0; root<Pn; root++)
      {
        int uLocalRows = yMap->getNumEntries(root);
        if(uLocalRows == 0) { continue; }

        // Compute the local TTM
      	local_tensor_type localResult;
      	if(X.getDistribution().ownNothing()) {
      	  std::vector<int> sz(ndims);
      	  for(int i=0; i<ndims; i++) {
      	    sz[i] = X.localExtent(i);
      	  }
      	  sz[n] = uLocalRows;
      	  localResult = local_tensor_type(sz);
      	}
        else
	  {
	   std::vector<int> I(localX.rank());
	   for(int i=0; i<(int)I.size(); i++) {
	     I[i] = (i != n) ? localX.extent(i) : uLocalRows;
      	  }
      	  localResult = local_tensor_type(I);

	  const std::size_t Unrows = (Utransp) ? localX.extent(n) : localResult.extent(n);
	  const std::size_t Uncols = (Utransp) ? localResult.extent(n) : localX.extent(n);
	  Kokkos::LayoutStride layout(Unrows, 1, Uncols, stride);
	  umv_type Aum(Uptr, layout);
	  TuckerOnNode::ttm(localX, n, Aum, localResult, Utransp);
        }

        // Combine the local results with a reduce operation
      	std::vector<ScalarType> sendBuf;
      	if(localResult.size() > 0){
      	  sendBuf.resize(localResult.size());
      	  Tucker::impl::copy_view_to_stdvec(localResult.data(), sendBuf);
      	}

      	ScalarType* recvBuf = nullptr;
      	if(localY.size() > 0){
      	  recvBuf = localYview_h.data();
      	}
        size_t count = localResult.size();
        assert(count <= std::numeric_limits<std::size_t>::max());

        if(count > 0) {
          MPI_Reduce_(sendBuf.data(), recvBuf, (int)count, MPI_SUM, root, comm);
        }

        if(Utransp){ Uptr += (uLocalRows*stride);}
        else{ Uptr += uLocalRows; }
      } // end for root

      Kokkos::deep_copy(localY.data(), localYview_h);
    }

  } // end if Pn != 1

  return Y;
}
#endif

}} // end namespace TuckerMpi::impl

#endif  // IMPL_TUCKERMPI_TTM_IMPL_HPP_
