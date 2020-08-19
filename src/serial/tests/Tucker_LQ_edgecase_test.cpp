#include <cmath>
#include "Tucker.hpp"

bool checkEqual(const double* arr1, const double* arr2, int nrows, int ncols)
{
    int ind = 0;
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++) {
        //std::cout << "matching:  arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
        if(std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) > 1e-10) {
          std::cout << arr1[r+c*nrows] << ", " << arr2[r+c] << std::endl;
          return false;
        }
        ind++;
      }
    }
  return true;
}

int main(){
  Tucker::SizeArray* size = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(4);
  Tucker::SizeArray* sizeA = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(4);
  int sizes[4] = {2,3,2,2};
  int sizesA[4] = {2,3,3,3};
  for(int i=0; i<4; i++){
    (*size)[i] = sizes[i];
    (*sizeA)[i] = sizesA[i];
  }
  Tucker::Tensor* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor>(*size);
  Tucker::Tensor* A = Tucker::MemoryManager::safe_new<Tucker::Tensor>(*sizeA);
  double* data = Y->data();
  int content[24] = {14,1,17,14,13,10,7,19,9,9,19,16,5,16,6,24,26,15,26,3,16,2,13,3};
  for(int i=0; i<24; i++)
    data[i] = content[i];
  double R1[9] = {-39.661064030103880, -33.332439064079679, -33.937566550870834,
                  0, -20.197735171038733,-13.604403099311346,
                  0,0,-25.063156084512997};
  double* dataA = A->data();
  int contentA[54] = {70,29,23,56,72,43,99,69,49,40,35,73,44,6,40,74,19,18,54,54,64,85,73,62,73,33,37,23,30,64,10,44,44,50,43,32,43,90,95,51,63,12,32,42,87,26,49,99,52,62,13,83,61,55};
  for(int i=0; i<54; i++)
    dataA[i] = contentA[i];
  double R1A[9] = {-236.317582926028,-192.825262664707,-191.521931798733,
                   0,-148.810006647022,-58.2536980555317,
                   0,0,119.586187759860};
  bool isEqual1 = false;
  // Tucker::Matrix* r = Tucker::computeLQ(Y, 1); 
  // isEqual1 = checkEqual(r->data(), R1, r->nrows(), r->ncols());
  // Tucker::MemoryManager::safe_delete<Tucker::Matrix>(r);
  Tucker::Matrix* r = Tucker::computeLQ(A, 1);
  std::cout << r->prettyPrint() << std::endl;
  isEqual1 = checkEqual(r->data(), R1A, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(r);
  if(!isEqual1){
     return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}