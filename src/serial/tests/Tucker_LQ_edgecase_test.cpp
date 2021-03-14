#include <cmath>
#include <limits>
#include "Tucker.hpp"

template <class scalar_t>
bool checkEqual(const scalar_t* arr1, const scalar_t* arr2, int nrows, int ncols)
{
    int ind = 0;
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++) {
        //std::cout << "matching:  arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
        if(std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) > 100 * std::numeric_limits<scalar_t>::epsilon() * std::abs(arr2[ind]) ) {
          std::cout.precision(std::numeric_limits<scalar_t>::max_digits10);
          std::cout << arr1[r+c*nrows] << ", " << arr2[ind] << std::endl;
          std::cout <<  " difference is " << std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) << std::endl;
          std::cout <<  " threshold is " <<100 * std::numeric_limits<scalar_t>::epsilon() * std::abs(arr2[ind]);
          return false;
        }
        ind++;
      }
    }
  return true;
}

int main(){

// Specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t; 
#else
  typedef double scalar_t;
#endif

  Tucker::SizeArray* size = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(4);
  Tucker::SizeArray* sizeA = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(4);
  Tucker::SizeArray* sizeB = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(3);
  int sizes[4] = {2,3,2,2};
  int sizesA[4] = {2,3,3,3};
  for(int i=0; i<4; i++){
    (*size)[i] = sizes[i];
    (*sizeA)[i] = sizesA[i];
  }
  (*sizeB)[0] = 3; (*sizeB)[1] = 9; (*sizeB)[2] = 2;
  Tucker::Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*size);
  Tucker::Tensor<scalar_t>* A = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*sizeA);
  Tucker::Tensor<scalar_t>* B = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*sizeB);
  int content[24] = {14,1,17,14,13,10,7,19,9,9,19,16,5,16,6,24,26,15,26,3,16,2,13,3};
  for(int i=0; i<24; i++)
    Y->data()[i] = content[i];
  int contentA[54] = {70,29,23,56,72,43,99,69,49,40,35,73,44,6,40,74,19,18,54,54,64,85,73,62,73,33,37,23,30,64,10,44,44,50,43,32,43,90,95,51,63,12,32,42,87,26,49,99,52,62,13,83,61,55};
  for(int i=0; i<54; i++){
    A->data()[i] = contentA[i];
    B->data()[i] = contentA[i];
  }
    
  scalar_t R1[9] = {-39.661064030103880, -33.332439064079679, -33.937566550870834,
                  0, -20.197735171038733,-13.604403099311346,
                  0,0,-25.063156084512997};
  
  scalar_t R1A[9] = {-236.317582926028,-192.825262664707,-191.521931798733,
                   0,-148.810006647022,-58.2536980555317,
                   0,0,119.586187759860};
  
  scalar_t R1B[54] = { -1.086047881080756e+02,  -1.046270629310778e+02,  -1.339351630199294e+02,  -1.405370818900854e+02,  -7.370761583765538e+01,  -1.262283204894962e+02,  -1.401595663061574e+02,  -1.232081958180720e+02,  -1.305375227645772e+02,   
 000000000000000       ,  -5.651705673879594e+01,  -4.256692419685490e+01,  -3.122982672458242e+01,  -1.263669200883330e+01,  1.636494334468578e+01 ,  -3.134480700013239e+01,  -6.389377916842577e+01,  -5.860927832361764e+00,   
 000000000000000       ,  000000000000000       ,  -5.084711467963679e+01,  2.242437250066056e+01 ,  -3.860697619722456e+01,  9.849238896028822e+00 ,  2.602992114491679e+01 ,  -5.802351956036775e+01,  -4.163370824250568e+01,   
 000000000000000       ,  000000000000000       ,  000000000000000       ,  -7.680608084340453e+01,  -5.215006711291323e+01,  -1.185389906217946e+01,  -3.117964081501874e+01,  -1.982172560141818e+01,  -4.511636913403473e+01,   
 000000000000000       ,  000000000000000       ,  000000000000000       ,  000000000000000       ,  -2.198575155931868e+01,  -3.496714410082862e+00,  2.215098060632534e+01 ,  -9.524107758615589e+00,  1.357496536116377e+00 ,  
 000000000000000       ,  000000000000000       ,  000000000000000       ,  000000000000000       ,  000000000000000       ,  -8.176202923923469e+00,  6.197282388946130e+00 ,  1.409684448392638e+00 ,  -2.401893945879391e+01, };

  bool isEqual1 = false;
  bool isEqual2 = false;
  bool isEqual3 = false;
  Tucker::Matrix<scalar_t>* r = Tucker::computeLQ(Y, 1); 
  //std::cout << r->prettyPrint() << std::endl;
  isEqual1 = checkEqual(r->data(), R1, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);

  r = Tucker::computeLQ(A, 1);
  //std::cout << r->prettyPrint() << std::endl;
  isEqual2 = checkEqual(r->data(), R1A, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);

  r = Tucker::computeLQ(B, 1);
  //std::cout << r->prettyPrint() << std::endl;
  isEqual3 = checkEqual(r->data(), R1B, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);
  if(!isEqual1 || !isEqual2 || !isEqual3){
     return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}