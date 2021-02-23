#include<random>
#include<chrono>
#include "TuckerMPI.hpp"
#include "Tucker_IO_Util.hpp"
int main(int argc, char* argv[]){
// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t;
  std::string filename = "input_files/tensor64_single.mpi"; 
#else
  typedef double scalar_t;
  std::string filename = "input_files/tensor64.mpi";
#endif

  // Initialize MPI
  MPI_Init(&argc,&argv);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  int n = 4;
  int seed = 234;
  scalar_t tol = 1e-8;
  scalar_t threshold = 1e-6;
  TuckerMPI::TuckerTensor<scalar_t>* fact = Tucker::MemoryManager::safe_new<TuckerMPI::TuckerTensor<scalar_t>>(n);
  Tucker::SizeArray* proc_grid_dims = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(n);
  (*proc_grid_dims)[0] = 1; (*proc_grid_dims)[1] = 1; (*proc_grid_dims)[2] = 2;  (*proc_grid_dims)[3] = 5; 
  Tucker::SizeArray* tensor_dims = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(n);
  (*tensor_dims)[0] = 10; (*tensor_dims)[1] = 20; (*tensor_dims)[2] = 10;  (*tensor_dims)[3] = 20; 
  Tucker::SizeArray* core_dims = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(n);
  (*core_dims)[0] = 5; (*core_dims)[1] = 10; (*core_dims)[2] = 5; (*core_dims)[3] = 10; 
  TuckerMPI::Tensor<scalar_t>* T = TuckerMPI::generateTensor<scalar_t>(seed, fact, proc_grid_dims, tensor_dims, core_dims, 1e-12);
  // std::cout << "core tensor is originally: ";
  // for(int i = 0; i < 4; i++){
  //   std::cout << fact->G->getGlobalSize().data()[i] << ", ";
  // }
  // for(int i=0; i< 8; i++){
  //   std::cout << T->getLocalTensor()->data()[i]<< ", ";
  // }
  std::cout << std::endl;
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(proc_grid_dims);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(tensor_dims);

  const TuckerMPI::TuckerTensor<scalar_t>* oldGramSolution = TuckerMPI::STHOSVD<scalar_t>(T, threshold, nullptr, true, false, false);
  const TuckerMPI::TuckerTensor<scalar_t>* newGramSolution = TuckerMPI::STHOSVD<scalar_t>(T, threshold, nullptr, false, false, false);
  const TuckerMPI::TuckerTensor<scalar_t>* LQSolution = TuckerMPI::STHOSVD<scalar_t>(T, threshold, nullptr, false, false, true);
  Tucker::MemoryManager::safe_delete(T);
  Tucker::MemoryManager::safe_delete(core_dims);

  bool isCoreEqual_old_newGram = Tucker::isApproxEqual(newGramSolution->G->getLocalTensor(), oldGramSolution->G->getLocalTensor(), tol);
  bool isCoreEqual_LQ_newGram = Tucker::isApproxEqual(newGramSolution->G->getLocalTensor(), LQSolution->G->getLocalTensor(), tol,false, true);
  if(!isCoreEqual_old_newGram){
    std::cout << "Local core not equal between old and new gram. Reported by processor " << rank << "." << std::endl;
  }
  if(!isCoreEqual_LQ_newGram){
    std::cout << "Local core not equal between LQ and new gram. Reported by processor " << rank << "." << std::endl;
  }
  bool isCoreEqual = isCoreEqual_LQ_newGram && isCoreEqual_old_newGram;
  int* compare_results = Tucker::MemoryManager::safe_new_array<int>(nprocs);
  MPI_Allgather(&isCoreEqual, 1, MPI_INT, compare_results, 1, MPI_INT, MPI_COMM_WORLD);
  for(int i=0; i<nprocs; i++){
    if(compare_results[i] == 0){
      Tucker::MemoryManager::safe_delete(fact);
      Tucker::MemoryManager::safe_delete_array<int>(compare_results, nprocs);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
  }
  for(int i=0; i<n; i++){
    bool isFactorMatrixEqual_old_newGram = Tucker::isApproxEqual(newGramSolution->U[i], oldGramSolution->U[i], tol, false, true);
    bool isFactorMatrixEqual_LQ_newGram = Tucker::isApproxEqual(newGramSolution->U[i], LQSolution->U[i], tol, false, true);
    if(!isFactorMatrixEqual_old_newGram){
      std::cout << "Factor matrix of mode "<< i <<" not equal between old and new gram. Reported by processor " << rank << "." << std::endl;
    }
    if(!isFactorMatrixEqual_LQ_newGram){
      std::cout << "Factor matrix of mode "<< i <<" not equal between LQ and new gram. Reported by processor " << rank << "." << std::endl;
    }
    bool isFactorMatrixEqual = isFactorMatrixEqual_old_newGram && isFactorMatrixEqual_LQ_newGram;
    MPI_Allgather(&isFactorMatrixEqual, 1, MPI_INT, compare_results, 1, MPI_INT, MPI_COMM_WORLD);
    for(int i=0; i<nprocs; i++){
      if(compare_results[i] == 0){
        Tucker::MemoryManager::safe_delete(fact);
        Tucker::MemoryManager::safe_delete_array<int>(compare_results, nprocs);
        MPI_Finalize();
        return EXIT_FAILURE;
      }
    }
  }
  Tucker::MemoryManager::safe_delete(fact);
  Tucker::MemoryManager::safe_delete_array<int>(compare_results, nprocs);
  MPI_Finalize();
  return EXIT_FAILURE;
}