#include <cmath>
#include "Tucker.hpp"

bool checkEqual(const double* arr1, const double* arr2, int nrows, int ncols)
{
    int ind = 0;
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++) {
        //std::cout << "matching:  arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
        if(std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) > 1e-10) {
          //std::cout << arr1[r+c*nrows] << ", " << arr2[r+c] << std::endl;
          return false;
        }
        ind++;
      }
    }
  return true;
}

int main(){
  Tucker::SizeArray* size = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(4);
  int sizes[4] = {2,3,2,2};
  for(int i=0; i<4; i++)
    (*size)[i] = sizes[i];
  Tucker::Tensor* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor>(*size);
  double* data = Y->data();
  int content[24] = {14,1,17,14,13,10,7,19,9,9,19,16,5,16,6,24,26,15,26,3,16,2,13,3};
  for(int i=0; i<24; i++)
    data[i] = content[i];
  double R1[9] = {-39.661064030103880, -33.332439064079679, -33.937566550870834,
                  0, -20.197735171038733,-13.604403099311346,
                  0,0,-25.063156084512997};
  bool isEqual1 = false;
  Tucker::Matrix* r = Tucker::computeLQ(Y, 1); 
  std::cout << "R1" << "(" << r->nrows() << "*" << r->ncols() <<") " << std::endl;
  for(int i=0; i<r->ncols()*r->nrows();i++ ){
    std::cout << r->data()[i] << ", "; 
  }
  std::cout << std::endl;
  isEqual1 = checkEqual(r->data(), R1, r->nrows(), r->ncols());
  if(!isEqual1){
     return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}