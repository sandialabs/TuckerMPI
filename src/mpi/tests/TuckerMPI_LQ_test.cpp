#include<cstdlib>
#include "TuckerMPI.hpp"
bool checkEqual(const double* arr1, const double* arr2, int nrows, int ncols)
{
    int ind = 0;
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++) {
        //std::cout << "matching:  arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
        if(abs(arr1[r+c*nrows]) - abs(arr2[ind]) > 1e-10) {
          //std::cout << arr1[r+c*nrows] << ", " << arr2[r+c] << std::endl;
          return false;
        }
        ind++;
      }
    }
  return true;
}

//three scenarios to test here:
//1. Each mode n processor fiber only has one processor in it. In this case NO redistribution is done
//and sequential LQ is called.
//2. Each mode n processor fiber has multiple processors AND n=N-1. In this case redistribution is done
//with each local unfolding being row major. Sequential LQ is not called.
//3. Each mode n processor fiber has multiple processors AND n!=N-1. In this case redistribution is done
//with each local unfolding being column major. Sequential LQ is not called.
int main(int argc, char* argv[])
{
    // Initialize MPI
  MPI_Init(&argc,&argv);

  // Get rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // Create a distribution object
  int ndims = 4;
  Tucker::SizeArray* sz =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*sz)[0] = 5; (*sz)[1] = 4; (*sz)[2] = 7; (*sz)[3] = 2;

  Tucker::SizeArray* nprocsPerDim1 =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*nprocsPerDim1)[0] = 1; (*nprocsPerDim1)[1] = 1; (*nprocsPerDim1)[2] = 5; (*nprocsPerDim1)[3] = 1;
  TuckerMPI::Distribution* dist1 =
    Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*sz,*nprocsPerDim1);

  Tucker::SizeArray* nprocsPerDim2 =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*nprocsPerDim2)[0] = 1; (*nprocsPerDim2)[1] = 1; (*nprocsPerDim2)[2] = 1; (*nprocsPerDim2)[3] = 5;
  TuckerMPI::Distribution* dist2 =
        Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*sz,*nprocsPerDim2);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(sz);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(nprocsPerDim1);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(nprocsPerDim2);
  
  double trueL0[25] = {-3.952328427648695e+02, -2.749012436313084e+02, -3.100197826244291e+02, -3.681171812094462e+02, -3.628822923638578e+02
                      ,0, -3.006281860537364e+02, -1.363683716566369e+02, -1.194735914650778e+02, -1.118232025369253e+02
                      ,0, 0, 2.572846703424420e+02, 1.045028152718186e+02, 0.386737482123126e+02
                      ,0, -0, 0, -2.164739324748169e+02, -0.592004686022506e+02
                      ,0, 0, -0, -0, -2.035427694113008e+02};
  double trueL2[49] = {-3.577121747997962e+02,-2.764205609030233e+02,-2.899845387092458e+02,-2.847736453393367e+02,-2.771529933402101e+02,-3.020836516410947e+02,-2.650426982337477e+02
                      ,0,-2.469082289233350e+02,-0.961503440379753e+02,-1.225828290814063e+02,-1.165218925687584e+02,-0.532824184560492e+02,-0.680566571808925e+02
                      ,0,0,1.791844821712107e+02,0.775773909769400e+02,0.052643901105738e+02,0.227546979075584e+02,0.331727632906067e+02
                      ,0,0,0,-1.678397132497413e+02,-0.647889028668548e+02,0.017470380540700e+02,-0.064155891946835e+02
                      ,0,0,-0,0,-1.971155780166274e+02,-0.085606757138384e+02,-0.358228357383247e+02
                      ,0,-0,0,-0,0,-1.968612143291617e+02,-0.296829164705747e+02
                      ,0,-0,0,0,-0,0,1.872504110189170e+02};
  double trueL3[4] = {-6.754916727836103e+02, -5.063141616396524e+02,
                      0,  -4.342832828041347e+02};

  // This tensor is used to test 
  TuckerMPI::Tensor* tensor1 =
    Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(dist1);
  TuckerMPI::Tensor* tensor2 =
    Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(dist2);      

  // Read the entries from a file
  std::string filename = "input_files/lq_data.mpi";
  TuckerMPI::importTensorBinary(filename.c_str(),tensor1);
  TuckerMPI::importTensorBinary(filename.c_str(),tensor2);
  bool t1l0Correct = false;
  bool t1l2Correct = false;
  bool t2l3Correct = false;
  Tucker::Matrix* L0 = TuckerMPI::LQ(tensor1, 0);
  if(rank == 0){
    // { 
    //   std::cout << "Mode 0 final result: " << std::endl; 
    //   for(int i=0; i< L0->nrows(); i++){
    //       for(int j=0; j<L0->ncols(); j++){
    //           std::cout << L0->data()[i+j*L0->nrows()] << ", ";
    //       }
    //       std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    // }
    t1l0Correct = checkEqual(L0->data(), trueL0, 5, 5);

    if(!t1l0Correct){
      Tucker::MemoryManager::safe_delete<TuckerMPI::Tensor>(tensor1);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
  }
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L0);

  Tucker::Matrix* L2 = TuckerMPI::LQ(tensor1, 2);
  if(rank == 0){
    // {    
    //   std::cout << "Mode 2 final result: " << std::endl; 
    //   for(int i=0; i< L2->nrows(); i++){
    //     for(int j=0; j<L2->ncols(); j++){
    //       std::cout << L2->data()[j+i*L2->nrows()] << ", ";
    //     }
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    // }
    t1l2Correct = checkEqual(L2->data(), trueL2, 4, 4);
    if(!t1l2Correct){
      Tucker::MemoryManager::safe_delete<TuckerMPI::Tensor>(tensor1);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
  }
  Tucker::MemoryManager::safe_delete<TuckerMPI::Tensor>(tensor1);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L2);

  Tucker::Matrix* L3 = TuckerMPI::LQ(tensor2, 3);
  if(rank == 0){
    t2l3Correct = checkEqual(L3->data(), trueL3, 2, 2);
    if(!t2l3Correct){
      Tucker::MemoryManager::safe_delete<TuckerMPI::Tensor>(tensor2);
      Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L3);
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    // {    
    //   std::cout << "Mode 3 final result: " << std::endl; 
    //   for(int i=0; i< L3->nrows(); i++){
    //     for(int j=0; j<L3->ncols(); j++){
    //       std::cout << L3->data()[i+j*L3->nrows()] << ", ";
    //     }
    //     std::cout << std::endl;
    //   }
    //   std::cout << std::endl;
    // }
  }
  Tucker::MemoryManager::safe_delete<TuckerMPI::Tensor>(tensor2);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L3);
  MPI_Finalize();
  return EXIT_SUCCESS;

}