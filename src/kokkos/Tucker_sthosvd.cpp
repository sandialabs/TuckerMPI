#include "Tucker_IO_Util.hpp"
#include "Tucker_BlasWrapper.hpp"
#include "init_args.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "KokkosBlas1_nrm2.hpp"
#include "compute_gram.hpp"
#include "ttm.hpp"
#include <fstream>
#include <iomanip>

// ------------------------------
namespace TuckerKokkos{
// ------------------------------

template<class ScalarType, class MemorySpace>
class Tensor
{
  static_assert(std::is_floating_point_v<ScalarType>, "");

  using view_type = Kokkos::View<ScalarType*, MemorySpace>;
  using exespace = typename view_type::execution_space;

public:
  Tensor() = default;
  Tensor(const Tucker::SizeArray & I) : I_(I.size())
  {
    // Copy the SizeArray
    for(int i=0; i<I.size(); i++) {
      if(I[i] < 0) {
	std::ostringstream oss;
	oss << "TuckerKokkos::Tensor(const SizeArray& I): I["
	    << i << "] = " << I[i] << " < 0.";
	throw std::length_error(oss.str());
      }
      I_[i] = I[i];
    }

    // Compute the total number of entries in this tensor
    const size_t numEntries = getNumElements();
    data_ = view_type("tensorData", numEntries);
  }

  int N() const{ return I_.size(); }
  const Tucker::SizeArray& size() const{ return I_; }

  int size(const int n) const{
    if(n < 0 || n >= N()) {
      std::ostringstream oss;
      oss << "Tucker::Tensor::size(const int n): n = "
	  << n << " is not in the range [0," << N() << ")";
      throw std::out_of_range(oss.str());
    }
    return I_[n];
  }

  size_t getNumElements() const{ return I_.prod(); }
  ScalarType norm2() const{ return ::KokkosBlas::nrm2(data_); }

  const view_type data() const{ return data_; }

  void print(int precision = 2) const{
    auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data_);

    // If this tensor doesn't have any entries, there's nothing to print
    size_t numElements = getNumElements();
    if(numElements == 0){
      return;
    }
    const ScalarType* dataPtr = v_h.data();
    for(size_t i=0; i<numElements; i++) {
      std::cout << "data[" << i << "] = " << std::setprecision(precision) << dataPtr[i] << std::endl;
    }
  }

  void initialize(){
    throw std::runtime_error("Tensor::initialize missing impl");
  }

  void rand(ScalarType a, ScalarType b){
    Kokkos::Random_XorShift64_Pool<exespace> pool(4543423);
    Kokkos::fill_random(data_, pool, a, b);
  }

private:
  view_type data_;
  Tucker::SizeArray I_;
};


template <class ScalarType, class MemorySpace>
void importTensorBinary(TuckerKokkos::Tensor<ScalarType, MemorySpace> & X,
			const char* filename)
{
  // Get the maximum file size we can read
  const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  std::ifstream ifs;
  ifs.open(filename, std::ios::in | std::ios::binary);
  assert(ifs.is_open());

  std::streampos begin, end, size;
  begin = ifs.tellg();
  ifs.seekg(0, std::ios::end);
  end = ifs.tellg();
  size = end - begin;
  //std::cout << "Reading " << size << " bytes...\n";
  size_t numEntries = X.getNumElements();
  assert(size == numEntries*sizeof(ScalarType));

  // Read the file
  auto view1d_d = X.data();
  auto view1d_h = Kokkos::create_mirror(view1d_d);
  ScalarType* data = view1d_h.data();
  ifs.seekg(0, std::ios::beg);
  ifs.read((char*)data,size);

  Kokkos::deep_copy(view1d_d, view1d_h);
  ifs.close();
}

template <class ScalarType, class MemorySpace>
void readTensorBinary(TuckerKokkos::Tensor<ScalarType, MemorySpace> & Y,
		      const char* filename)
{
  std::ifstream inStream(filename);
  std::string temp;
  int nfiles = 0;
  while(inStream >> temp) { nfiles++; }
  inStream.close();
  if(nfiles != 1) {
    throw std::runtime_error("readTensorBinary hardwired for one file only for now");
  }
  importTensorBinary(Y, temp.c_str());
}



template<class ScalarType, class MemorySpace>
class TuckerTensor {
public:
  TuckerTensor(const int ndims) : N(ndims)
  {
    assert(ndims > 0);

    // U = MemoryManager::safe_new_array<Matrix<ScalarType>*>(N);
    // eigenvalues = MemoryManager::safe_new_array<ScalarType*>(N);
    // singularValues = MemoryManager::safe_new_array<ScalarType*>(N);
    // G = 0;

    // for(int i=0; i<N; i++) {
    //   singularValues[i] = 0;
    //   eigenvalues[i] = 0;
    //   U[i] = 0;
    // }
  }

  auto getFactorMatrix(int n){
    return Kokkos::subview(U, n, Kokkos::ALL, Kokkos::ALL);
  }

private:
  int N;
  Tensor<ScalarType, MemorySpace> G;
  Kokkos::View<ScalarType***, Kokkos::LayoutLeft, MemorySpace> U;
  // ScalarType** eigenvalues;
  // ScalarType** singularValues;

// private:
//   TuckerTensor(const TuckerTensor<ScalarType, MemorySpace>& tt);
};

template<class ScalarType, class MemorySpace>
auto computeGram(Tensor<ScalarType, MemorySpace> * Y, const int n)
{
  const int nrows = Y->size(n);
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> S_d("S", nrows, nrows);
  auto S_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), S_d);
  computeGramHost(Y, n, S_h.data(), nrows);
  Kokkos::deep_copy(S_d, S_h);
  return S_d;
}

template<class ScalarType, class ... Props>
auto computeEigenvalues(Kokkos::View<ScalarType**, Props...> G,
			const bool flipSign)
{
  using view_type = Kokkos::View<ScalarType**, Props...>;
  using mem_space = typename view_type::memory_space;
  static_assert(std::is_same_v< typename view_type::array_layout, Kokkos::LayoutLeft>);

  const int nrows = G.extent(0);
  auto G_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G);
  Kokkos::View<ScalarType*, mem_space> eigenvalues_d("EIG", nrows);
  auto eigenvalues_h = Kokkos::create_mirror_view(eigenvalues_d);

  char jobz = 'V';
  char uplo = 'U';
  int lwork = 8*nrows;
  ScalarType* work = Tucker::MemoryManager::safe_new_array<ScalarType>(lwork);
  int info;
  Tucker::syev(&jobz, &uplo, &nrows, G_h.data(), &nrows,
	       eigenvalues_h.data(), work, &lwork, &info);

  // Check the error code
  if(info != 0){
    std::cerr << "Error: invalid error code returned by dsyev (" << info << ")\n";
  }

  // The user will expect the eigenvalues to be sorted in descending order
  // LAPACK gives us the eigenvalues in ascending order
  for(int esubs=0; esubs<nrows-esubs-1; esubs++) {
    ScalarType temp = eigenvalues_h[esubs];
    eigenvalues_h[esubs] = eigenvalues_h[nrows-esubs-1];
    eigenvalues_h[nrows-esubs-1] = temp;
  }

  // Sort the eigenvectors too
  ScalarType* Gptr = G_h.data();
  const int ONE = 1;
  for(int esubs=0; esubs<nrows-esubs-1; esubs++) {
    Tucker::swap(&nrows, Gptr+esubs*nrows, &ONE, Gptr+(nrows-esubs-1)*nrows, &ONE);
  }

  if(flipSign){
    for(int c=0; c<nrows; c++)
    {
      int maxIndex=0;
      ScalarType maxVal = std::abs(Gptr[c*nrows]);
      for(int r=1; r<nrows; r++)
      {
        ScalarType testVal = std::abs(Gptr[c*nrows+r]);
        if(testVal > maxVal) {
          maxIndex = r;
          maxVal = testVal;
        }
      }

      if(Gptr[c*nrows+maxIndex] < 0) {
        const ScalarType NEGONE = -1;
	Tucker::scal(&nrows, &NEGONE, Gptr+c*nrows, &ONE);
      }
    }
  }

  Tucker::MemoryManager::safe_delete_array<ScalarType>(work,lwork);

  Kokkos::deep_copy(G, G_h);
  Kokkos::deep_copy(eigenvalues_d, eigenvalues_h);
  return eigenvalues_d;
}

template <class ScalarType, class MemorySpace>
int findEigvalsCountToKeep(Kokkos::View<ScalarType*, MemorySpace> eigvals,
			   const ScalarType thresh)
{
  auto eigvals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigvals);

  int nrows = eigvals_h.extent(0);
  int numEvecs=nrows;
  ScalarType sum = 0;
  for(int i=nrows-1; i>=0; i--) {
    sum += std::abs(eigvals_h[i]);
    if(sum > thresh) {
      break;
    }
    numEvecs--;
  }
  return numEvecs;
}

template <class ScalarType, class MemorySpace>
auto STHOSVD(Tensor<ScalarType, MemorySpace> & X,
	     const ScalarType epsilon,
	     bool useQR = false,
	     bool flipSign = false)
{
  const int ndims = X.N();

  using factor_type = TuckerKokkos::TuckerTensor<ScalarType, MemorySpace>;
  factor_type factorization(ndims);

  // Compute the threshold
  const ScalarType tensorNorm = X.norm2();
  const ScalarType thresh = epsilon*epsilon*tensorNorm/ndims;
  std::cout << "\tAutoST-HOSVD::Tensor Norm: "
	    << std::sqrt(tensorNorm) << "...\n";
  std::cout << "\tAutoST-HOSVD::Relative Threshold: "
	    << thresh << "...\n";

  Tensor<ScalarType, MemorySpace> * Y = &X;
  Tensor<ScalarType, MemorySpace> temp;
  for (int n=0; n<ndims; n++)
  {
    std::cout << "\tAutoST-HOSVD::Starting Gram(" << n << ")...\n";

    std::cout << " \n ";
    std::cout << "\tAutoST-HOSVD::Gram(" << n << ") \n";
    auto S = computeGram(Y, n);
    auto S_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), S);
    for (int i=0; i<S_h.extent(0); ++i){
      for (int j=0; j<S_h.extent(1); ++j){
	std::cout << S_h(i,j) << "  ";
      }
      std::cout << " \n ";
    }

    std::cout << " \n ";
    std::cout << "\tAutoST-HOSVD::Starting Evecs(" << n << ")...\n";
    auto eigvals = computeEigenvalues(S, flipSign);
    // need to copy back to S_h because of the reordering
    Kokkos::deep_copy(S_h, S);
    auto eigvals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigvals);
    for (int i=0; i<S.extent(0); ++i){ std::cout << eigvals_h(i) << "  "; }
    std::cout << " \n ";
    const int numEvecs = findEigvalsCountToKeep(eigvals, thresh);
    Kokkos::fence();


    std::cout << " \n ";
    using eigvec_view_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace>;
    eigvec_view_t eigVecs("eigVecs", Y->size(n), numEvecs);
    auto eigVecs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigVecs);
    const int nToCopy = Y->size(n)*numEvecs;
    const int ONE = 1;
    Tucker::copy(&nToCopy, S_h.data(), &ONE, eigVecs_h.data(), &ONE);
    for (int i=0; i<eigVecs_h.extent(0); ++i){
      for (int j=0; j<eigVecs_h.extent(1); ++j){
	std::cout << eigVecs_h(i,j) << "  ";
      }
      std::cout << " \n ";
    }
    Kokkos::deep_copy(eigVecs, eigVecs_h);
    Kokkos::fence();


    std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    std::cout << " \n ";
    temp = ttm(Y, n, eigVecs, true);
    temp.print();
    Kokkos::fence();

    Y = &temp;
    size_t nnz = Y->getNumElements();
    std::cout << "Local tensor size after STHOSVD iteration "
	      << n << ": " << Y->size() << ", or ";
  }

  // factorization->G = const_cast<Tensor<ScalarType>*>(Y);
  // factorization->total_timer_.stop();
  return factorization;
}

// ------------------------------
} // namespace TuckerKokkos
// ------------------------------

int main(int argc, char* argv[])
{
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif

  Kokkos::initialize();
  {
    //
    // parsing
    //
    const std::string paramfn =
      Tucker::parseString(argc, (const char**)argv,
			  "--parameter-file", "paramfile.txt");
    const std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
    InputArgs args = parse_input_file<scalar_t>(fileAsString);
    int checkArgs = check_args(args);
    std::cout << "Argument checking: passed" << std::endl;
    print_args(args);

    chech_array_sizes(args);
    std::cout << "Array sizes checking: passed" << std::endl;

    Tucker::SizeArray* I_dims = Tucker::stringParseSizeArray(fileAsString, "Global dims");
    args.nd = I_dims->size();
    std::cout << "The global dimensions of the tensor to be scaled or compressed\n";
    std::cout << "- Global dims = " << *I_dims << std::endl << std::endl;

    Tucker::SizeArray* R_dims = nullptr;
    if (!args.boolAuto) {
      R_dims = Tucker::stringParseSizeArray(fileAsString, "Ranks");
      std::cout << "Global dimensions of the desired core tensor\n";
      std::cout << "Not used if \"Automatic rank determination\" is enabled\n";
      std::cout << "- Ranks = " << *R_dims << std::endl << std::endl;
    }

    //
    // reading data
    //
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    TuckerKokkos::Tensor<scalar_t, memory_space> X(*I_dims);
    //X.rand(-15.56, 22.13);
    TuckerKokkos::readTensorBinary(X, args.in_fns_file.c_str());
    // auto v = X.data();
    // for (int i=0; i<v.extent(0); ++i){
    //   std::cout << v(i) << " \n";
    // }

    //
    // compute
    //
    // FIXME: Compute statistics is missing
    // FIXME: Perform preprocessing is missing
    if(args.boolSTHOSVD) {
      auto f = TuckerKokkos::STHOSVD(X, args.tol, args.boolUseLQ);
    }

    // // Free memory
    Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(I_dims);
    if(R_dims) Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(R_dims);
    //Tucker::MemoryManager::printMaxMemUsage();
  }

  Kokkos::finalize();
  return 0;
}
