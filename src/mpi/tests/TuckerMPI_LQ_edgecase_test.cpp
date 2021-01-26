#include<cstdlib>
#include "TuckerMPI.hpp"
#include<cmath>

bool checkEqual(const double* arr1, const double* arr2, int nrows, int ncols)
{
    int ind = 0;
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++){
        //std::cout << "matching:  arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
        if(std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) > 1e-10) {
          std::cout << "mismatch :" << "arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
          return false;
        }
        ind++;
      }
    }
  return true;
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc,&argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  int root = 0;
  std::string inputFilename = "input_files/lq_edgecase_input.mpi";
  std::string outputFilename = "input_files/lq_edgecase_output.txt";
  int ndims = 4;
  Tucker::SizeArray* tensorSizeArray =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*tensorSizeArray)[0] = 96; 
  (*tensorSizeArray)[1] = 6; 
  (*tensorSizeArray)[2] = 6; 
  (*tensorSizeArray)[3] = 6;
  Tucker::SizeArray* nprocsPerDim = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*nprocsPerDim)[0] = 1; 
  (*nprocsPerDim)[1] = 2; 
  (*nprocsPerDim)[2] = 2; 
  (*nprocsPerDim)[3] = 2;
  TuckerMPI::Distribution* dist = 
  Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*tensorSizeArray,*nprocsPerDim);
  TuckerMPI::Tensor* tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(dist);
  TuckerMPI::importTensorBinary(inputFilename.c_str(),tensor);
  Tucker::Matrix* trueL =Tucker::importMatrix(outputFilename.c_str());
  int compareResultBuff;

  Tucker::Matrix* L = TuckerMPI::LQ(tensor, 0);
  // if(rank == 0) std::cout << L->prettyPrint();
  if(rank == 0){
    compareResultBuff = checkEqual(L->data(), trueL->data(), 96, 96);
  }
  MPI_Bcast(&compareResultBuff, 1, MPI_INT, root, MPI_COMM_WORLD);
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
  tensor = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(dist);
  TuckerMPI::importTensorBinary(inputFilename.c_str(),tensor);
  L = TuckerMPI::LQ(tensor, 0);
  if(rank == 0){
    compareResultBuff = checkEqual(L->data(), trueL->data(), 96, 96);
  }
  MPI_Bcast(&compareResultBuff, 1, MPI_INT, root, MPI_COMM_WORLD);
  Tucker::MemoryManager::safe_delete(L);
  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(trueL);
  if(compareResultBuff != 1){
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}