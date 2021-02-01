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
        if(std::abs(std::abs(arr1[r+c*nrows]) - std::abs(arr2[ind])) > 100 * std::numeric_limits<scalar_t>::epsilon()) {
          std::cout << arr1[r+c*nrows] << ", " << arr2[r+c] << std::endl;
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
  int sizes[4] = {3,2,3,2};
  for(int i=0; i<4; i++)
    (*size)[i] = sizes[i];
  Tucker::Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*size);
  scalar_t* data = Y->data();
  int content[36] = {1,4,4,2,1,3,3,2,1,4,3,2,5,2,4,6,3,1,6,3,1,5,2,4,4,3,2,3,2,1,2,1,3,1,4,4};
  for(int i=0; i<36; i++)
    data[i] = content[i];
    //exact solution in column major, lower triangular.
  scalar_t R0[9] = {-13.490737563232042, -7.708992893275450, -6.967743576614350,
                  0, 5.154748157905345, 4.323337164694807,
                  0, 0, 5.172939706870563};
  scalar_t R1[4] = {-13.453624047073712, -11.595388681455795,
                  0, 6.822533351033680};
  scalar_t R2[9] = {-11.747340124470732,-7.661308776828736,-8.001811389132236,
                  0, 5.225356239156038,5.491616429686285,
                  0, 0, -6.619151265982165};
  scalar_t R3[4] = {-13.453624047073712, -9.811482730462597,
                  0, 9.205151092178458};
  bool isEqual0 = false;
  bool isEqual1 = false;
  bool isEqual2 = false;
  bool isEqual3 = false;
  Tucker::Matrix<scalar_t>* r = Tucker::computeLQ(Y, 0); 
  std::cout << "r is " << r->nrows() << " by " << r->ncols() << std::endl;
  std::cout << r->prettyPrint();
  isEqual0 = checkEqual(r->data(), R0, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);

  r = Tucker::computeLQ(Y, 1); 
  std::cout << "R1" << "(" << r->nrows() << "*" << r->ncols() <<") " << std::endl;
  std::cout << r->prettyPrint();
  isEqual1 = checkEqual(r->data(), R1, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);

  r = Tucker::computeLQ(Y, 2); 
  std::cout << "R2: " << std::endl;
  std::cout << r->prettyPrint();
  isEqual2 = checkEqual(r->data(), R2, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);

  r = Tucker::computeLQ(Y, 3); 
  std::cout << "R3: " << std::endl;
  std::cout << r->prettyPrint();
  isEqual3 = checkEqual(r->data(), R3, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);
  Tucker::MemoryManager::safe_delete(Y);
  Tucker::MemoryManager::safe_delete(size);
  if(!isEqual0 || !isEqual1 || !isEqual2 || !isEqual3){
     return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}