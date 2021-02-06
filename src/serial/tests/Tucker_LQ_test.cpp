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
  int sizes[4] = {3,2,3,2};
  for(int i=0; i<4; i++)
    (*size)[i] = sizes[i];
  Tucker::Tensor* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor>(*size);
  int content[36] = {1,4,4,2,1,3,3,2,1,4,3,2,5,2,4,6,3,1,6,3,1,5,2,4,4,3,2,3,2,1,2,1,3,1,4,4};
  for(int i=0; i<36; i++)
    Y->data()[i] = content[i];
    //exact solution in column major, lower triangular.
  double R0[9] = {-13.490737563232042, -7.708992893275450, -6.967743576614350,
                  0, 5.154748157905345, 4.323337164694807,
                  0, 0, 5.172939706870563};
  double R1[4] = {-13.453624047073712, -11.595388681455795,
                  0, 6.822533351033680};
  double R2[9] = {-11.747340124470732,-7.661308776828736,-8.001811389132236,
                  0, 5.225356239156038,5.491616429686285,
                  0, 0, -6.619151265982165};
  double R3[4] = {-13.453624047073712, -9.811482730462597,
                  0, 9.205151092178458};
  double R4[36] = {-7.874007874011811,-5.080005080007620,-2.667002667004001,-7.366007366011051,-3.048003048004572,-6.604006604009907,-6.858006858010288,-5.588005588008382,-3.683003683005525,
                   0,3.491926171484267,3.852204268542168,1.311781789287741,0.720556194115801,2.420329780235128,2.623563578575482,1.607394586873710,0.655890894643871,
                   0,0,0.218217890235992,1.382046638161285,0.436435780471985,0.290957186981322,2.764093276322570,-0.436435780471987,-1.600264528397281,
                   0,0,0,-0.333333333333332,0.000000000000000,0.666666666666667,-0.666666666666664,2.000000000000000,2.333333333333331};

  bool isEqual0, isEqual1, isEqual2, isEqual3, isEqual4 = false;
  Tucker::Matrix* r = Tucker::computeLQ(Y, 0); 
  // std::cout << "r is " << r->nrows() << " by " << r->ncols() << std::endl;
  // std::cout << r->prettyPrint();
  isEqual0 = checkEqual(r->data(), R0, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(r);

  r = Tucker::computeLQ(Y, 1); 
  // std::cout << "R1" << "(" << r->nrows() << "*" << r->ncols() <<") " << std::endl;
  // std::cout << r->prettyPrint();
  isEqual1 = checkEqual(r->data(), R1, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(r);

  r = Tucker::computeLQ(Y, 2); 
  // std::cout << "R2: " << std::endl;
  // std::cout << r->prettyPrint();
  isEqual2 = checkEqual(r->data(), R2, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(r);

  r = Tucker::computeLQ(Y, 3); 
  // std::cout << "R3: " << std::endl;
  // std::cout << r->prettyPrint();
  isEqual3 = checkEqual(r->data(), R3, r->nrows(), r->ncols());
  Tucker::MemoryManager::safe_delete(r);
  Tucker::MemoryManager::safe_delete(Y);
  Tucker::MemoryManager::safe_delete(size);

  //Test for when Yn is tall and skinny
  size = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(3);
  (*size)[0] = 9; 
  (*size)[1] = 2; 
  (*size)[2] = 2;
  Y = Tucker::MemoryManager::safe_new<Tucker::Tensor>(*size);
  for(int i=0; i<36; i++)
    Y->data()[i] = content[i];
  r = Tucker::computeLQ(Y, 0); 
  isEqual4 = checkEqual(r->data(), R4, r->nrows(), r->ncols());

  if(!isEqual0 || !isEqual1 || !isEqual2 || !isEqual3 || !isEqual4){
     return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}