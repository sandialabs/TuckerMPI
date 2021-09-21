#include<cstdlib>
#include "TuckerMPI.hpp"
#include<cmath>

template <class scalar_t>
bool checkEqual(const scalar_t* arr1, const scalar_t* arr2, int nrows, int ncols)
{
    int ind = 0;
    std::cout.precision(std::numeric_limits<scalar_t>::max_digits10);
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++){
        if(std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) > 1000000 * std::numeric_limits<scalar_t>::epsilon() * std::abs(arr2[ind])) {
          std::cout << "mismatch :" << "arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
          std::cout << "diff: " << std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) << std::endl;
          std::cout << "threshold: " << 1000000 * std::numeric_limits<scalar_t>::epsilon() * std::abs(arr2[ind]) << std::endl;
          return false;
        }
        ind++;
      }
    }
  return true;
}

//Testing the edge case where each processor owns a tall and skinny submatrix the unfolded tensor.
int main(int argc, char* argv[])
{
// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t;
  std::string inputFilename = "input_files/lq_edgecase_input_single.mpi"; 
#else
  typedef double scalar_t;
  std::string inputFilename = "input_files/lq_edgecase_input.mpi";
#endif

  MPI_Init(&argc,&argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  int root = 0;
  
  int ndims = 4;
  Tucker::SizeArray* tensorSizeArray =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*tensorSizeArray)[0] = 96; 
  (*tensorSizeArray)[1] = 6; 
  (*tensorSizeArray)[2] = 6; 
  (*tensorSizeArray)[3] = 6;
  std::string outputFilename = "input_files/lq_edgecase_output.txt";
  Tucker::Matrix<scalar_t>* trueL =Tucker::importMatrix<scalar_t>(outputFilename.c_str());
  int compareResultBuff;
  TuckerMPI::Distribution* dist;
  TuckerMPI::Tensor<scalar_t>* tensor;
  Tucker::SizeArray* nprocsPerDim;
  Tucker::Matrix<scalar_t>* L;


  if (np == 8)
  {
    nprocsPerDim = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
    (*nprocsPerDim)[0] = 1; 
    (*nprocsPerDim)[1] = 2; 
    (*nprocsPerDim)[2] = 2; 
    (*nprocsPerDim)[3] = 2;
    dist = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*tensorSizeArray,*nprocsPerDim);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);
    TuckerMPI::importTensorBinary(inputFilename.c_str(),tensor);

    L = TuckerMPI::LQ<scalar_t>(tensor, 0, false);
    // std::cout <<"rank: " << rank << std::endl;
    compareResultBuff = checkEqual(L->data(), trueL->data(), 96, 96);
    Tucker::MemoryManager::safe_delete(L);
    Tucker::MemoryManager::safe_delete(tensor);
    if(compareResultBuff != 1){
      MPI_Finalize();
      return EXIT_FAILURE;
    }

    nprocsPerDim = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
    (*nprocsPerDim)[0] = 2; 
    (*nprocsPerDim)[1] = 1; 
    (*nprocsPerDim)[2] = 2; 
    (*nprocsPerDim)[3] = 2;
    dist = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*tensorSizeArray,*nprocsPerDim);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);
    TuckerMPI::importTensorBinary(inputFilename.c_str(),tensor);
    L = TuckerMPI::LQ(tensor, 0, false);
    compareResultBuff = checkEqual(L->data(), trueL->data(), 96, 96);
    Tucker::MemoryManager::safe_delete(L);
    Tucker::MemoryManager::safe_delete(tensor);
    Tucker::MemoryManager::safe_delete(trueL);
    if(compareResultBuff != 1){
      MPI_Finalize();
      return EXIT_FAILURE;
    }
  }
  else if(np == 9) {
    nprocsPerDim = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
    (*nprocsPerDim)[0] = 1; 
    (*nprocsPerDim)[1] = 1; 
    (*nprocsPerDim)[2] = 3; 
    (*nprocsPerDim)[3] = 3;
    dist = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*tensorSizeArray,*nprocsPerDim);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);
    TuckerMPI::importTensorBinary(inputFilename.c_str(),tensor);

    L = TuckerMPI::LQ<scalar_t>(tensor, 0, false);
    std::cout <<"rank: " << rank << std::endl;
    compareResultBuff = checkEqual(L->data(), trueL->data(), 96, 96);
    Tucker::MemoryManager::safe_delete(L);
    Tucker::MemoryManager::safe_delete(tensor);
    if(compareResultBuff != 1){
      MPI_Finalize();
      return EXIT_FAILURE;
    }

    nprocsPerDim = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
    (*nprocsPerDim)[0] = 3; 
    (*nprocsPerDim)[1] = 1; 
    (*nprocsPerDim)[2] = 1; 
    (*nprocsPerDim)[3] = 3;
    dist = Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*tensorSizeArray,*nprocsPerDim);
    tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);
    TuckerMPI::importTensorBinary(inputFilename.c_str(),tensor);

    L = TuckerMPI::LQ<scalar_t>(tensor, 0, false);
    std::cout <<"rank: " << rank << std::endl;
    compareResultBuff = checkEqual(L->data(), trueL->data(), 96, 96);
    Tucker::MemoryManager::safe_delete(L);
    Tucker::MemoryManager::safe_delete(tensor);
    if(compareResultBuff != 1){
      MPI_Finalize();
      return EXIT_FAILURE;
    }
  }
  
  MPI_Finalize();
  return EXIT_SUCCESS;
}