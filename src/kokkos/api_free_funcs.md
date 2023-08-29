# Tucker Public API

Header File: `<Tucker_create_mirror.hpp>`

```cpp
namespace Tucker{

// ------------------------------------------
// Overloads accepting a TuckerOnNode::Tensor
// ------------------------------------------

template<class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror(const ::TuckerOnNode::Tensor<ScalarType, Properties...> & tensor);

// (A)
template<class SpaceT, class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror_tensor_and_copy(const SpaceT & space,
						                         const ::TuckerOnNode::Tensor<ScalarType, Properties...> & tensor);

// (B)
template<class SpaceT, class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror_tensor_and_copy(const SpaceT & space,
						                         const ::TuckerOnNode::Tensor<ScalarType, Properties...> & tensor);


// ----------------------------------------------
// Overloads accepting a TuckerOnNode::MetricData
// ----------------------------------------------

template<class ScalarType, class MemorySpace>
[[nodiscard]] auto create_mirror(::TuckerOnNode::MetricData<ScalarType, MemorySpace> d);


// -----------------------------------------------------------------------
// Overloads accepting a TuckerMpi::Tensor (must define TUCKER_ENABLE_MPI)
// -----------------------------------------------------------------------

// (A)
template<class SpaceT, class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror_tensor_and_copy(const SpaceT & space,
						                         ::TuckerMpi::Tensor<ScalarType, Properties...> tensor);

// (B)
template<class SpaceT, class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror_tensor_and_copy(const SpaceT & space,
						                         ::TuckerMpi::Tensor<ScalarType, Properties...> tensor);

}//end namespace Tucker
```

- (A): Enable if `SpaceT::memory_space` != `TuckerOnNode::Tensor<ScalarType, Properties...>::traits::memory_space`.

- (B): Enable if `SpaceT::memory_space` == `TuckerOnNode::Tensor<ScalarType, Properties...>::traits::memory_space`.


<br>

-----------------


## TuckerMpi::sthosvd

### Interface

```cpp
namespace TuckerMpi{

template <class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto sthosvd(Method method,
                           ::TuckerMpi::Tensor<ScalarType, Properties...> tensor,
                           TruncatorType && truncator,
                           const std::vector<int> & modeOrder,
                           bool flipSign);

}//end namespace TuckerMpi
```

### Parameters

- `method`: Specifies which algorithm to use, currently only accepting `TuckerMpi::Method::NewGram`.

- `tensor`: Data tensor to operate on.

- `truncator`: Function object accepting the mode and a Kokkos view with the gram eigenvalues, and returning the rank along the specify mode. Signature should meet:

  ```cpp
  std::size_t truncator(int mode, auto eigenvalues);
  ```
  See `Tucker::create_core_tensor_truncator` for more information on the truncator.

- `modeOrder`: Specifies the order of the modes to operate on.

- `flipSign`: Only applicable to gram. If `true`, the sign of an eigenvector is flipped if its max element is negative.

### Constraints

- Tensor layout have to be `Kokkos::LayoutLeft`.
- Tensor value type have to be `double`.

### Return value

- A std::pair containing an instance of the `Tucker::TuckerTensor` class and an instance of the `TuckerOnNode::TensorGramEigenvalues`.

### Example usage

```cpp
std::vector<int> extents  = {33,44,65,21};
std::vector<int> procGrid = {2,1,2,2};
TuckerMpi::Tensor<double> T(extents, procGrid);
// read data into tensor or fill somehow

const auto method = TuckerMpi::Method::NewGram;
auto truncator = Tucker::create_core_tensor_truncator(T, {}, 1e-2, mpiRank);
const std::vector<int> order = {0,1,2,3};
auto [tuckTensor, eigvals] = TuckerMpi::sthosvd(method, T, truncator, order, false);

auto coreT = tuckTensor.coreTensor(); // get the core tensor
// coreT is a TuckerMpi::Tensor that lives in the same memory space as the tensor

auto f0 = tuckTensor.factorMatrix(0); // get factor matrix for mode 0
// f0 is a rank-2 view that lives in the same memory space as the tensor

auto v0 = eigvals[0];
// v0 is a rank-1 with the eigenvalues computed by gram for mode 0
```


<br>

-----------------


## Tucker::create_core_tensor_truncator

The function `create_core_tensor_truncator` is used to create a functor that knows how to "truncate" the core tensor when doing sthosvd either operating on eigenvalues via a tolerance or using a fixed core tensor extents.
If `fixedCoreTensorRanks` is provided then it is used and `tol` is disregarded.

- Note that this work for both on-node (kokkos only) and distributed (MPI+Kokkos).

### Description 

```cpp
namespace Tucker{

template <class TensorType, class ScalarType>
[[nodiscard]] auto create_core_tensor_truncator(TensorType dataTensor,
						                        const std::optional<std::vector<int>> & fixedCoreTensorRanks,
						                        ScalarType tol,
						                        int mpiRank = 0)
{
  return [=](std::size_t mode, auto eigenValues) -> std::size_t
  {
    if (fixedCoreTensorRanks){
      (void) eigenValues; // unused
      return (*fixedCoreTensorRanks)[mode];
    }
    else{
      (void) mode; // unused

      const auto rank = dataTensor.rank();
      const ScalarType norm = dataTensor.frobeniusNormSquared();
      const ScalarType threshold  = tol*tol*norm/rank;
      if (mpiRank==0){
	     std::cout << "  AutoST-HOSVD::Tensor Norm: " << std::sqrt(norm) << "...\n";
	     std::cout << "  AutoST-HOSVD::Relative Threshold: " << threshold << "...\n";
      }
      auto eigVals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigenValues);
      return impl::count_eigvals_using_threshold(eigVals_h, threshold);
    }
  };
}

}//end namespace Tucker
```

### Usage

```cpp
// ...
int mpiRank;
MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

std::vector<int> extents  = {33,44,65,21};
std::vector<int> procGrid = {2,1,2,2};
TuckerMpi::Tensor<double> T(extents, procGrid);
// read data into tensor or fill somehow

// 1. let's create a truncator that computes the truncation for the core tensor that is based on a tolerance (1e-2)
auto truncator = Tucker::create_core_tensor_truncator(X, {}, 1e-2, mpiRank);
auto [tuckTensor, eigvals] = TuckerMpi::sthosvd(method, T truncator, {}, false /*flipSign*/);
auto coreT = tuckTensor.coreTensor();
// coreT has extents computed based on the eigenvalues and tol=1e-2

// 2. let's pass a fixed desired extents for the core tensor
std::vector<int> targetCoreExts = {2,2,2,2};
auto truncator = Tucker::create_core_tensor_truncator(X, targetCoreExts, 1e-2, mpiRank);
auto [tuckTensor, eigvals] = TuckerMpi::sthosvd(method, T truncator, {}, false /*flipSign*/);
// coreT has extents = 2,2,2,2 as specified
```


<br>

-----------------


## TuckerMpi::compute_gram

Computes the gram matrix (https://en.wikipedia.org/wiki/Gram_matrix) of `tensor` along the mode `mode`.

### Interface

```cpp
namespace TuckerMpi{

template<class ScalarType, class ...Properties>
[[nodiscard]] auto compute_gram(Tensor<ScalarType, Properties...> tensor,
                                const std::size_t mode);

}//end namespace TuckerMpi
```

### Constraints

Parameter `tensor` must have `Kokkos::LayoutLeft` and `double` value type.

### Preconditions

The tensor must be non-empty.

### Returns

A rank-2 Kokkos::View with LayoutLeft, same scalar type and memory_space as the input tensor.

### Side effects

If the tensor is non-trivially distributed (i.e. multiple MPI processes), the implementation uses these MPI collectives: `MPI_Alltoallv` and `MPI_Allreduce`.

### Post-conditions

The input `tensor` is unmodified.

<br>

-----------------

## TuckerMpi::ttm

Compute the ttm of `tensor` and `Umatrix` for mode `n`.

### Interface

```cpp
namespace TuckerMpi{

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
[[nodiscard]] auto ttm(Tensor<ScalarType, TensorProperties...> tensor,
        		       int n,
        		       Kokkos::View<ScalarType**, ViewProperties...> Umatrix,
        		       bool Utransp,
        		       std::size_t nnz_limit);

}//end namespace TuckerMpi
```

### Constraints

- Parameter `tensor` must have `Kokkos::LayoutLeft`.
- Parameter  `tensor` and `Umatrix` must have the same memory space.

### Returns

A new tensor with the result.
The *type* of the returned tensor is the same as the input tensor.


<br>

-----------------


## TuckerMpi::compute_slice_metrics

Computes the metrics specified in `metrics` for the tensor `tensor` along the mode `mode`.

### Interface

```cpp
namespace TuckerMpi{

template <std::size_t n, class ScalarType, class ...Properties>
[[nodiscard]] auto compute_slice_metrics(const int mpiRank,
					                     Tensor<ScalarType, Properties...> tensor,
					                     const int mode,
					                     const std::array<Tucker::Metric, n> & metrics);

}//end namespace TuckerMpi                              
```           

### Constraints

Parameter `tensor` must have `Kokkos::LayoutLeft` and `double` value type.

### Preconditions

- Parameter `mode` must be non-negative.

- Valid choices for the entries in `metrics` are

  ```cpp
  namespace Tucker{
    enum class Metric {
      MIN, MAX, SUM, NORM1, NORM2, MEAN, VARIANCE
    };
  }
  ```

### Returns

A instance of `TuckerOnNode::MetricData` with the target metrics computed.

### Example Usage

```cpp
// ...
int mpiRank;
MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

std::vector<int> extents  = {33,44,65,21};
std::vector<int> procGrid = {2,1,2,2};
TuckerMpi::Tensor<double> T(extents, procGrid);
// read data into tensor or fill somehow

int scaleMode = 0;
auto metricsData = TuckerMpi::compute_slice_metrics(mpiRank, T, scaleMode, Tucker::defaultMetrics);
```


<br>

-----------------


### TuckerMpi::normalize_tensor

```cpp
template <class ScalarType, class MetricMemSpace, class ...Props>
[[nodiscard]] auto normalize_tensor(const int mpiRank,
                				    ::TuckerMpi::Tensor<ScalarType, Props...> & tensor,
                				    const TuckerOnNode::MetricData<ScalarType, MetricMemSpace> & metricData,
                				    const std::string & scalingType,
                				    const int scaleMode,
                				    const ScalarType stdThresh)
```

Normalizes the tensor `tensor` along mode `scaleMode` using the metrics given in `metricData` and according to `scalingType`. If needed, `stdThresh` is used.

**Constraints**: `tensor` must have `Kokkos::LayoutLeft` and `double` value type

**Returns**: A std::pair containing the scales and shifts used for normalizing. The returned scales and shifts are both Kokkos rank-1 views with the same memory space as the input tensor.

#### Example Usage

```cpp
// ...
int mpiRank;
MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

std::vector<int> extents  = {33,44,65,21};
std::vector<int> procGrid = {2,1,2,2};
TuckerMpi::Tensor<double> T(extents, procGrid)
// read data into tensor or fill somehow

int scaleMode = 0;
auto metricsData = TuckerMpi::compute_slice_metrics(mpiRank, T, scaleMode, Tucker::defaultMetrics);
auto [scales, shifts] = TuckerMpi::normalize_tensor(mpiRank, T, metricsData, "Max", scaleMode, 0.5);
```
