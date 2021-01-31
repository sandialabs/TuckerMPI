#include <cassert>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
#include "Tucker_IO_Util.hpp"
#include "Tucker.hpp"

int main(int argc, char* argv[])
{
  int one = 1;
  int negOne = -1;
  std::string paramfn = Tucker::parseString(argc, (const char**)argv,
    "--parameter-file", "paramfile.txt");
  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
  int YNrows                             = Tucker::stringParse<int>(fileAsString, "Nrows", 0);
  int YNcols                             = Tucker::stringParse<int>(fileAsString, "Ncols", 0);
  int nb                                = Tucker::stringParse<int>(fileAsString, "NB", 1);
  int avgIteration                      = Tucker::stringParse<int>(fileAsString, "AvgIteration", 10);
  std::cout << "NB: " << nb << std::endl;
  std::cout << "YNrows: " << YNrows << std::endl;
  std::cout << "YNcols: " << YNcols << std::endl;
  std::cout << "AvgIteration: " << avgIteration << std::endl;

  Tucker::Timer qrWorkSpaceQueryTimer;
  Tucker::Timer qrTimer;
  Tucker::Timer ATATimer;
  Tucker::Timer AATTimer;

  int info;
  //dgeqr
  Tucker::Matrix* Y = Tucker::MemoryManager::safe_new<Tucker::Matrix>(YNrows, YNcols);
  int sizeOfY = YNrows*YNcols;
  Y->rand();
  Tucker::Matrix* YCopy = Tucker::MemoryManager::safe_new<Tucker::Matrix>(YNrows, YNcols);
  double* work = Tucker::MemoryManager::safe_new_array<double>(1);
  qrWorkSpaceQueryTimer.start();
  work = Tucker::MemoryManager::safe_new_array<double>(1);
  //T has to have space for at least 5 doubles, see dgeqr.
  double* T = Tucker::MemoryManager::safe_new_array<double>(5);
  Tucker::dgeqr_(&YNrows, &YNcols, Y->data(), &YNrows, T, &negOne, work, &negOne, &info);
  int lwork = work[0];
  int TSize = T[0];
  Tucker::MemoryManager::safe_delete_array<double>(work, 1);
  Tucker::MemoryManager::safe_delete_array<double>(T, 5);
  qrWorkSpaceQueryTimer.stop();
  std::cout << "qr lwork:" << lwork << " TSize: " << TSize << " Size_MAX: " << SIZE_MAX <<  std::endl;
  work = Tucker::MemoryManager::safe_new_array<double>(lwork);
  T = Tucker::MemoryManager::safe_new_array<double>(TSize);
  for(int i=0; i<avgIteration; i++){
    dcopy_(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
    qrTimer.start();
    Tucker::dgeqr_(&YNrows, &YNcols, YCopy->data(), &YNrows, T, &TSize, work, &lwork, &info);
    qrTimer.stop();
  }
  double avgQrTime = qrTimer.duration() / avgIteration;
  std::cout << "avgQrTime: " << avgQrTime <<  std::endl;
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(Y);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(YCopy);
  Tucker::MemoryManager::safe_delete_array<double>(work, lwork);
  Tucker::MemoryManager::safe_delete_array<double>(T, TSize);

  int ANrows = YNrows;
  int ANcols = YNcols;
  Tucker::Matrix* A = Tucker::MemoryManager::safe_new<Tucker::Matrix>(ANrows, ANcols);
  A->rand();
  //compute A'*A
  char uplo = 'U';
  char trans = 'T';
  double alpha = 1;
  double beta = 0;
  Tucker::Matrix* S = Tucker::MemoryManager::safe_new<Tucker::Matrix>(ANcols, ANcols);
  for(int i=0; i<avgIteration; i++){
    ATATimer.start();
    Tucker::dsyrk_(&uplo, &trans, &ANcols, &ANrows, &alpha,
        A->data(), &ANrows, &beta, S->data(), &ANcols);
    ATATimer.stop();
  }
  double avgATATime = ATATimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(S);
  //comput A*A'
  Tucker::Matrix* S1 = Tucker::MemoryManager::safe_new<Tucker::Matrix>(ANrows, ANrows);
  S1->rand();
  trans = 'N';
  for(int i=0; i<avgIteration; i++){
    AATTimer.start();
    Tucker::dsyrk_(&uplo, &trans, &ANrows, &ANcols, &alpha,
        A->data(), &ANrows, &beta, S1->data(), &ANrows);
    AATTimer.stop();
  }
  double avgAATTime = AATTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(A);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(S1);

  std::cout << "work space query takes: " << qrWorkSpaceQueryTimer.duration() << " seconds. \n";
  std::cout << "dgeqr takes: " << avgQrTime << " seconds. \n";
  std::cout << "dsyrk ATA takes: " << avgATATime << " seconds. \n";
  std::cout << "dsyrk AAT takes: " << avgAATTime << " seconds. \n";
  return EXIT_SUCCESS;
}
