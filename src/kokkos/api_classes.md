# Tucker Public API

## Classes and "containers"

### TuckerMpi::Tensor

```cpp
template<class ScalarType, class ...Properties> class Tensor;
```

#### Traits

```cpp
namespace TuckerMpi{
namespace impl{

template<class Enable, class ScalarType, class ...Properties>
struct TensorTraits;

template<class ScalarType> struct TensorTraits<void, ScalarType>{
  using memory_space       = typename Kokkos::DefaultExecutionSpace::memory_space;
  using execution_space    = Kokkos::DefaultExecutionSpace;
  using onnode_tensor_type = TuckerOnNode::Tensor<ScalarType, memory_space>;
  using value_type         = typename onnode_tensor_type::traits::data_view_type::value_type;
  using onnode_layout      = typename onnode_tensor_type::traits::array_layout;
};

template<class ScalarType, class MemSpace>
struct TensorTraits<
  std::enable_if_t< Kokkos::is_memory_space_v<MemSpace> >, ScalarType, MemSpace >
{
  using memory_space       = MemSpace;
  using execution_space    = typename memory_space::execution_space;
  using onnode_tensor_type = TuckerOnNode::Tensor<ScalarType, memory_space>;
  using value_type         = typename onnode_tensor_type::traits::data_view_type::value_type;
  using onnode_layout      = typename onnode_tensor_type::traits::array_layout;
};
}//end namespace impl
```

#### Class API

```cpp
template<class ScalarType, class ...Properties>
class Tensor
{
  using dims_view_type            = Kokkos::View<int*>;
  using dims_host_view_type       = typename dims_view_type::HostMirror;
  using dims_const_view_type      = typename dims_view_type::const_type;
  using dims_host_const_view_type = typename dims_host_view_type::const_type;

public:
  using traits = impl::TensorTraits<void, ScalarType, Properties...>;

public:
  // -------------------------------------------------
  // Regular constructors, destructor, and assignment
  // -------------------------------------------------
  Tensor() = default;
  ~Tensor() = default;

  explicit Tensor(const Distribution & dist);
  Tensor(const std::vector<int>& extents, const std::vector<int>& procs);

  Tensor(const Tensor& o) = default;
  Tensor(Tensor&&) = default;
  Tensor& operator=(const Tensor& o);
  Tensor& operator=(Tensor&& o);

  // ---------------------------------------------------
  // copy/move constr, assignment for compatible Tensor
  // ---------------------------------------------------
  template<class ST, class ... PS> Tensor(const Tensor<ST,PS...> & o);
  template<class ST, class ... PS> Tensor& operator=(const Tensor<ST,PS...> & o);
  template<class ST, class ... PS> Tensor(Tensor<ST,PS...> && o);
  template<class ST, class ... PS> Tensor& operator=(Tensor<ST,PS...> && o);

  // ---------------------------------------------------
  // methods
  // ---------------------------------------------------
  int rank() const
  auto localTensor();
  const Distribution & getDistribution() const;
  dims_const_view_type globalDimensions() const
  dims_const_view_type localDimensions() const;
  dims_host_const_view_type globalDimensionsOnHost() const;
  dims_host_const_view_type localDimensionsOnHost() const;
  int globalExtent(int n) const;
  int localExtent(int n) const;
  size_t localSize() const;
  size_t globalSize() const;
  auto frobeniusNormSquared() const;
}
} // end namespace TuckerMpi
```

#### Usage and semantics

Another tensor is compatible if (a) it is an empty tensor, or (b) has the same distribution

```cpp
TuckerMpi::Tensor<double> T(distribution)
// if not space template is provided, it uses the memory space associated with default exe space

TuckerMpi::Tensor<double, Kokkos::HostSpace> T;
// specify to be on host

TuckerMpi::Tensor<double, Kokkos::CudaSpace> T;
// specify to be on cuda (must be )

TuckerMpi::Tensor<double> T;
// constructs an empty tensor, no allocations made

std::vector<int> extents  = {33,44,65,21};
std::vector<int> procGrid = {2,1,2,2};
TuckerMpi::Tensor<double> T(exts, procs)
// creates a rank-4 tensor with extents where procGrid specify the MPI ranks distribution for each axis

TuckerMpi::Tensor<double> T(distribution);
// allocates according to `ditribution` object

TuckerMpi::Tensor<double> T2 = T;
/* shallow copy, modifying the values of T2 also modifies the values in T */

TuckerMpi::Tensor<double> T1;
TuckerMpi::Tensor<double> T2(distribution);
T1 = T2; /* ok: assigning to an empty tensor */

TuckerMpi::Tensor<double> T1(d1);
TuckerMpi::Tensor<double> T2(d2);
T1 = T2; /* NOT ok, throws because d1 and d2 are different distributions */

TuckerMpi::Tensor<double> T1(d1);
TuckerMpi::Tensor<const double> T2 = T1;
/* ok: CANNOT modify T2*/
```


<br>

-----------------



### Tucker::TuckerTensor

```cpp
namespace Tucker{

template<class CoreTensorType>
struct TuckerTensorTraits
{
  using core_tensor_type     = CoreTensorType;
  using value_type           = typename core_tensor_type::traits::value_type;
  using memory_space         = typename core_tensor_type::traits::memory_space;
  using factors_store_view_t = Kokkos::View<value_type*, Kokkos::LayoutLeft, memory_space>;
};

template<class CoreTensorType>
class TuckerTensor
{
  // the slicing info is stored on the host
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

public:
  using traits = TuckerTensorTraits<CoreTensorType>;

private:
  template<class FactorsViewType>
  TuckerTensor(typename traits::core_tensor_type coreTensor,
	             FactorsViewType factors,
	             slicing_info_view_t slicingInfo);

public:
  ~TuckerTensor() = default;

  TuckerTensor(const TuckerTensor& o) = default;
  TuckerTensor(TuckerTensor&&) = default;
  TuckerTensor& operator=(const TuckerTensor&) = default;
  TuckerTensor& operator=(TuckerTensor&&) = default;

  // ----------------------------------------
  // copy/move constr, assignment for compatible TuckerTensor
  // ----------------------------------------
  template<class LocalArg> TuckerTensor(const TuckerTensor<LocalArg> & o);
  template<class LocalArg> TuckerTensor& operator=(const TuckerTensor<LocalArg> & o);
  template<class LocalArg> TuckerTensor(TuckerTensor<LocalArg> && o);
  template<class LocalArg> TuckerTensor& operator=(TuckerTensor<LocalArg> && o);

  //----------------------------------------
  // methods
  // ----------------------------------------
  int rank() const;

  typename traits::core_tensor_type coreTensor();

  auto factorMatrix(int mode);
};

}// end namespace Tucker
```

- IMPORTANT: this class has private constructors because a user is not allowed to instantiate this directly.
Only the function `auto ttensot = TuckerMpi::sthosvd(...)` is allowed to internally construct and return an instance of the TuckerTensor class above. And users can only *use* its public methods.
The reason for this is that while the API for querying the core tensor and factor matrices are clear and solid, users should not know how the actual object is constructed. Originally, this class was fully private so users would only need to know that the return type of calling `sthosvd` is a object that exposes a certain API.
However, we thought that making it fully private was a bit too much so making the consturctors private was kind of a compromise.

#### Example usage

```cpp
// ...
int mpiRank;
MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

std::vector<int> extents  = {33,44,65,21};
std::vector<int> procGrid = {2,1,2,2};
TuckerMpi::Tensor<double> T(extents, procGrid)
// read data into tensor or fill somehow

const auto method = TuckerMpi::Method::NewGram;
auto [tuckTensor, eigvals] = TuckerMpi::sthosvd(method, T /*, some other args */);

// tuckTensor is an instance of the TuckerTensor class that we can use to extract things

auto f0 = tuckTensor.factorMatrix(0); // get factor matrix for mode 0
// f0 is a rank-2 view that lives in the same memory space as the tensor

auto coreT = tuckTensor.coreTensor(); // get the core tensor
// coreT is a TuckerMpi::Tensor that lives in the same memory space as the tensor
```



<br>

-----------------


### TuckerOnNode::TensorGramEigenvalues

```cpp
namespace TuckerOnNode{

template<
  class ScalarType,
  class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space
>
class TensorGramEigenvalues
{
  // the slicing info is stored on the host
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

private:
  template<class EigvalsViewType>
  TensorGramEigenvalues(EigvalsViewType eigvals,
			            slicing_info_view_t slicingInfo);

public:
  TensorGramEigenvalues();
  ~TensorGramEigenvalues() = default;

  TensorGramEigenvalues(const TensorGramEigenvalues& o) = default;
  TensorGramEigenvalues(TensorGramEigenvalues&&) = default;
  TensorGramEigenvalues& operator=(const TensorGramEigenvalues&) = default;
  TensorGramEigenvalues& operator=(TensorGramEigenvalues&&) = default;

  // -------------------------------------------------------------------
  // copy/move constr, assignment for compatible TensorGramEigenvalues
  // -------------------------------------------------------------------
  template<class ... LocalArgs> TensorGramEigenvalues(const TensorGramEigenvalues<LocalArgs...> & o);
  template<class ... LocalArgs> TensorGramEigenvalues& operator=(const TensorGramEigenvalues<LocalArgs...> & o);
  template<class ... LocalArgs> TensorGramEigenvalues(TensorGramEigenvalues<LocalArgs...> && o);
  template<class ... LocalArgs> TensorGramEigenvalues& operator=(TensorGramEigenvalues<LocalArgs...> && o);

  //----------------------------------------
  // methods
  // ----------------------------------------
  int rank() const;
  auto operator[](int mode);
};

} // end namespace TuckerOnNode
```

- The class `TensorGramEigenvalues` was not present in the original code, and has been introduced to store the eigenvalues when doing sthosvd via Gram. Why are we doing this?
Because the original code, when doing sthosvd, was storing the core tensor, factors and eigenvalues from the Gram *all* inside the TuckerTensor class. However, when using QR instead of Gram, there are no eigenvalues so the methods inside Tucker are not applicable.
Therefore, we decided to separate things: in our new code, the TuckerTensor class (see above) stores only the core tensor and the factor matrices, and the `TensorGramEigenvalues` stores the eigenvalues.


- Note that it has private constructors because users are not allowed to instantiate this directly. The reason is similar to the one provided for the `TuckerTensor` class above.

- subscript operator gives you a rank-1 "view" of the eigenvalues for a specific mode. Note that users don't need to know *how* we store the eigenvalues, just how to query them. In practice, the eigenvalues are stored in a single 1D Kokkos view and what we call "slicing info" is what is used to know how to "slice" the view to only get the part needed.

#### Example usage

```cpp
std::vector<int> extents  = {33,44,65,21};
std::vector<int> procGrid = {2,1,2,2};
TuckerMpi::Tensor<double> T(extents, procGrid)
// read data into tensor or fill somehow

const auto method = TuckerMpi::Method::NewGram;
auto [tuckTensor, eigvals] = TuckerMpi::sthosvd(method, T /*, some other args */);

// eigvals is an instance of the TensorGramEigenvalues class
// get the eigenvalues for mode 0
auto v = eigvals[0];
// v is a rank-1 with the eigenvalues computed by gram
```




<br>

-----------------


### TuckerOnNode::MetricData

```cpp
namespace TuckerOnNode {

template<class ScalarType, class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
class MetricData;

}//end namespace TuckerOnNode
```

#### Class API

```cpp
namespace TuckerOnNode {

template<
  class ScalarType,
  class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
class MetricData
{

public:
  using map_t = Kokkos::UnorderedMap<Tucker::Metric, int, MemorySpace>;
  using HostMirror = MetricData<ScalarType, Kokkos::HostSpace>;

  MetricData() = default;

private:
  template<class MapType, class ValuesType>
  MetricData(MapType map, ValuesType values);

  template<std::size_t n>
  MetricData(const std::array<Tucker::Metric, n> & metrics, const int numValues);

public:
  std::size_t numMetricsStored() const;
  KOKKOS_FUNCTION bool contains(Tucker::Metric key) const;
  KOKKOS_FUNCTION auto get(Tucker::Metric key) const;
};

}//end namespace TuckerOnNode
```

A class for storing metrics.

Only used in drivers during the "Compute statistics" step.
