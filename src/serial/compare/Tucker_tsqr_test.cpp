#include<cmath>
#include "Tucker_IO_Util.hpp"
#include "Tucker.hpp"

template <class scalar_t>
bool checkEqual(const scalar_t* arr1, const scalar_t* arr2, int nrows, int ncols)
{
    int ind = 0;
    for(int c=0; c<ncols; c++) {
      for(int r=0; r<nrows; r++) {
        //std::cout << "matching:  arr1["<< r << ", " << c<< "]: " << arr1[r+c*nrows] << ", arr2[" << ind << "]: " << arr2[ind] << std::endl;
        if(std::abs(arr1[r+c*nrows] - arr2[ind]) > 1e-10) {
          std::cout << arr1[r+c*nrows] << ", " << arr2[r+c] << std::endl;
          return false;
        }
        ind++;
      }
    }
  return true;
}

int main(int argc, char* argv[])
{

// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t; 
#else
  typedef double scalar_t;
#endif

  Tucker::Timer* tsqr_timer = Tucker::MemoryManager::safe_new<Tucker::Timer>();
  Tucker::Timer* reorganize_timer = Tucker::MemoryManager::safe_new<Tucker::Timer>();
  Tucker::Timer* new_reorganize_timer = Tucker::MemoryManager::safe_new<Tucker::Timer>();
  Tucker::Timer* qr_timer = Tucker::MemoryManager::safe_new<Tucker::Timer>();
  Tucker::Timer* dcopy_timer = Tucker::MemoryManager::safe_new<Tucker::Timer>();

  std::string paramfn = Tucker::parseString(argc, (const char**)argv,
      "--parameter-file", "paramfile.txt");
  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
  int n                                 = Tucker::stringParse<int>(fileAsString, "LQ on mode ", 1);
  Tucker::SizeArray* Ydims = Tucker::stringParseSizeArray(fileAsString, "Tensor dims");
  bool printResult = Tucker::stringParse<bool>(fileAsString, "Print result", false);
  Tucker::Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*Ydims);
  Y->rand();
  std::cout << "Tensor dims: " << *Ydims << std::endl;
  std::cout << "LQ on mode " << n << "\n" << std::endl; 

  //////////////////////////////////////////////////////////
  // Reorganizing the whole unfolding and then compute LQ //
  //////////////////////////////////////////////////////////
  std::cout << "Time for reorganizing the whole unfolding first: " << std::endl;
  int YnTransposeNcols = Y->size(n);
  int YnTransposeNrows = 1;
  for(int i=0; i<Y->N(); i++){
    if(i != n){
      YnTransposeNrows *= Y->size(i);
    }
  }
  //number of column major submatrices in YnTranspose
  int nmats = 1;
  for(int i=n+1; i<Y->N(); i++) {
    nmats *= Y->size(i);
  }
  Tucker::Matrix<scalar_t>* R = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(YnTransposeNrows, YnTransposeNcols);
  Tucker::Matrix<scalar_t>* newR = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(YnTransposeNrows, YnTransposeNcols);
  reorganize_timer->start();
  Tucker::combineColumnMajorBlocks<scalar_t>(Y, R, n, 0, nmats, 1);
  reorganize_timer->stop();
  //std::cout << "R: " << R->prettyPrint()<< std::endl;
  new_reorganize_timer->start();
  Tucker::combineColumnMajorBlocks<scalar_t>(Y, newR, n, 0, nmats, 2);
  new_reorganize_timer->stop();
  if(!checkEqual(newR->data(), R->data(), YnTransposeNrows, YnTransposeNcols)){
    std::cout << "NOT EQUAL !!!!" << std::endl;
  }
  //std::cout << "newR: " << newR->prettyPrint()<< std::endl;
  std::cout << "Reorganize time: " << reorganize_timer->duration() << std::endl;
  std::cout << "new Reorganize time: " << new_reorganize_timer->duration() << std::endl;

  qr_timer->start();
  scalar_t * work = Tucker::MemoryManager::safe_new_array<scalar_t>(1);
  scalar_t * TforGeqr = Tucker::MemoryManager::safe_new_array<scalar_t>(5);
  int negOne = -1;
  int info = 1;
  Tucker::geqr(&YnTransposeNrows, &YnTransposeNcols, R->data(), &YnTransposeNrows, TforGeqr, &negOne, work, &negOne, &info);
  int lwork = work[0];
  int TSize = TforGeqr[0];
  Tucker::MemoryManager::safe_delete_array(work, 1);
  Tucker::MemoryManager::safe_delete_array(TforGeqr, 5);
  work = Tucker::MemoryManager::safe_new_array<scalar_t>(lwork);
  TforGeqr = Tucker::MemoryManager::safe_new_array<scalar_t>(TSize);    
  Tucker::geqr(&YnTransposeNrows, &YnTransposeNcols, R->data(), &YnTransposeNrows, TforGeqr, &TSize, work, &lwork, &info);
  Tucker::MemoryManager::safe_delete_array(work, lwork);
  Tucker::MemoryManager::safe_delete_array(TforGeqr, TSize);
  qr_timer->stop();
  std::cout << "qr time on reorganized data: " << qr_timer->duration() << std::endl;
  if(printResult) std::cout << "R of R: " << R->prettyPrint() << std::endl;
  Tucker::MemoryManager::safe_delete(R);

  ////////////////////////////////////////////////////////////
  // Reorganize the top n submatrices and then compute TSQR //
  // If submatrices are tall and skinny, compute TSQR       //
  // without any reorganization.                            //
  ////////////////////////////////////////////////////////////
  std::cout << "original algorithm timing: " << std::endl;
  Tucker::Matrix<scalar_t>* L = Tucker::computeLQ(Y, n);
  if(printResult) std::cout << L->prettyPrint() << std::endl;
  Tucker::MemoryManager::safe_delete(L);

  return EXIT_SUCCESS;
}