#include<cassert>
#include<cstdlib>
#include<cmath>
#include "Tucker_IO_Util.hpp"
#include "Tucker.hpp"

int main(int argc, char* argv[])
{
  //int a = mkl_get_num_threads();
  int one = 1;
  int negOne = -1;
  std::string paramfn = Tucker::parseString(argc, (const char**)argv,
    "--parameter-file", "paramfile.txt");
  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
  int nrows                             = Tucker::stringParse<int>(fileAsString, "Nrows", 0);
  int ncols                             = Tucker::stringParse<int>(fileAsString, "Ncols", 0);
  int nb                                = Tucker::stringParse<int>(fileAsString, "NB", 1);
  int avgIteration                      = Tucker::stringParse<int>(fileAsString, "AvgIteration", 10);
  std::cout << "NB: " << nb << std::endl;
  std::cout << "Nrows: " << nrows << std::endl;
  std::cout << "Ncols: " << ncols << std::endl;
  std::cout << "AvgIteration: " << avgIteration << std::endl;

  Tucker::Timer transposeTimer;
  Tucker::Timer qrfWorkSpaceQueryTimer;
  Tucker::Timer lqfWorkSpaceQueryTimer;
  Tucker::Timer qrWorkSpaceQueryTimer;
  Tucker::Timer lqWorkSpaceQueryTimer;
  Tucker::Timer qrfTimer;
  Tucker::Timer lqfTimer;
  Tucker::Timer qrtTimer;
  Tucker::Timer qrTimer;
  Tucker::Timer lqtTimer;
  Tucker::Timer lqTimer;

  Tucker::Matrix* Y = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrows, ncols);
  int sizeOfY = nrows*ncols;
  Y->rand();
  Tucker::Matrix* YCopy = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrows, ncols);

  int info;
  //this tau is for both dgeqrf and dgelqf.
  double* tau = Tucker::MemoryManager::safe_new_array<double>(std::min(nrows, ncols));
  double* work = Tucker::MemoryManager::safe_new_array<double>(1);

  //dgeqrf
  //workspace query
  qrfWorkSpaceQueryTimer.start();
  Tucker::dgeqrf_(&nrows, &ncols, Y->data(), &nrows, tau, work, &negOne, &info);
  qrfWorkSpaceQueryTimer.stop();
  int lwork = work[0];
  Tucker::MemoryManager::safe_delete_array<double>(work, 1);
  work = Tucker::MemoryManager::safe_new_array<double>(lwork);
  //query done
  for(int i=0; i<avgIteration; i++){
    dcopy_(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
    qrfTimer.start();
    Tucker::dgeqrf_(&nrows, &ncols, YCopy->data(), &nrows, tau, work, &lwork, &info);
    qrfTimer.stop();
  }
  double avgQrfTime = qrfTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete_array<double>(work, lwork);
  std::cout << "qrf lwork: " << lwork << std::endl;
  
  //dgeqrt
  double* T = Tucker::MemoryManager::safe_new_array<double>(nb*ncols);
  work = Tucker::MemoryManager::safe_new_array<double>(nb*ncols);
  for(int i=0; i<avgIteration; i++){
    dcopy_(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
    qrtTimer.start();
    Tucker::dgeqrt_(&nrows, &ncols, &nb, YCopy->data(), &nrows, T, &nb, work, &info);
    qrtTimer.stop();
  }
  double avgQrtTime = qrtTimer.duration() /avgIteration;
  Tucker::MemoryManager::safe_delete_array<double>(work, nb*ncols);
  Tucker::MemoryManager::safe_delete_array<double>(T, nb*ncols);

  //dgeqr
  qrWorkSpaceQueryTimer.start();

  work = Tucker::MemoryManager::safe_new_array<double>(1);
  //T has to have space for at least 5 doubles, see dgeqr.
  T = Tucker::MemoryManager::safe_new_array<double>(5);
  Tucker::dgeqr_(&nrows, &ncols, Y->data(), &nrows, T, &negOne, work, &negOne, &info);
  lwork = work[0];
  int TSize = T[0];
  Tucker::MemoryManager::safe_delete_array<double>(work, 1);
  Tucker::MemoryManager::safe_delete_array<double>(T, 5);
  qrWorkSpaceQueryTimer.stop();
  std::cout << "qr lwork:" << lwork << " TSize: " << TSize << std::endl;
  work = Tucker::MemoryManager::safe_new_array<double>(lwork);
  T = Tucker::MemoryManager::safe_new_array<double>(TSize);
  for(int i=0; i<avgIteration; i++){
    dcopy_(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
    qrTimer.start();
    Tucker::dgeqr_(&nrows, &ncols, YCopy->data(), &nrows, T, &TSize, work, &lwork, &info);
    qrTimer.stop();
  }
  double avgQrTime = qrTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(YCopy);
  Tucker::MemoryManager::safe_delete_array<double>(work, lwork);
  Tucker::MemoryManager::safe_delete_array<double>(T, TSize);


  //Get YTranspose
  transposeTimer.start();
  Tucker::Matrix* YTranspose = Y->getTranspose();
  transposeTimer.stop();
  Tucker::Matrix* YTransposeCopy = Tucker::MemoryManager::safe_new<Tucker::Matrix>(YTranspose->nrows(), YTranspose->ncols());

  //dgelqf
  //workspace query
  lqfWorkSpaceQueryTimer.start();
  work = Tucker::MemoryManager::safe_new_array<double>(1);
  Tucker::dgelqf_(&ncols, &nrows, YTranspose->data(), &ncols, tau, work, &negOne, &info);
  lwork = work[0];
  lqfWorkSpaceQueryTimer.stop();
  std::cout << "lqf lwork: " << lwork << std::endl;
  Tucker::MemoryManager::safe_delete_array<double>(work, 1);
  work = Tucker::MemoryManager::safe_new_array<double>(lwork);
  //query done
  for(int i=0; i<avgIteration; i++){
    dcopy_(&sizeOfY, YTranspose->data(), &one, YTransposeCopy->data(), &one);
    lqfTimer.start();
    Tucker::dgelqf_(&ncols, &nrows, YTransposeCopy->data(), &ncols, tau, work, &lwork, &info);
    lqfTimer.stop();
  }
  double avgLqfTime = lqfTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete_array<double>(work, lwork);  
  Tucker::MemoryManager::safe_delete_array<double>(tau, std::min(nrows, ncols));

  //dgelqt
  T = Tucker::MemoryManager::safe_new_array<double>(nb*nrows);
  work = Tucker::MemoryManager::safe_new_array<double>(nb*nrows);
  for(int i=0; i<avgIteration; i++){
    dcopy_(&sizeOfY, YTranspose->data(), &one, YTransposeCopy->data(), &one);
    lqtTimer.start();
    Tucker::dgelqt_(&ncols, &nrows, &nb, YTransposeCopy->data(), &ncols, T, &nb, work, &info);
    lqtTimer.stop();
  }
  double avgLqtTime = lqtTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete_array<double>(T, nb*nrows);
  Tucker::MemoryManager::safe_delete_array<double>(work, nb*nrows);

  //dgelq
  lqWorkSpaceQueryTimer.start();
  work = Tucker::MemoryManager::safe_new_array<double>(1);
  T = Tucker::MemoryManager::safe_new_array<double>(5);
  Tucker::dgelq_(&ncols, &nrows, YTranspose->data(), &ncols, T, &negOne, work, &negOne, &info);
  lwork = work[0];
  TSize = T[0];
  Tucker::MemoryManager::safe_delete_array<double>(work, 1);
  Tucker::MemoryManager::safe_delete_array<double>(T, 5);
  lqWorkSpaceQueryTimer.stop();
  std::cout << " lq lwork:" << lwork << " TSize: " << TSize << std::endl;
  work = Tucker::MemoryManager::safe_new_array<double>(lwork);
  T = Tucker::MemoryManager::safe_new_array<double>(TSize);
  for(int i=0; i<avgIteration; i++){
    dcopy_(&sizeOfY, YTranspose->data(), &one, YTransposeCopy->data(), &one);
    lqTimer.start();
    Tucker::dgelq_(&ncols, &nrows, YTransposeCopy->data(), &ncols, T, &TSize, work, &lwork, &info);
    lqTimer.stop();
  }
  double avgLqTime = lqTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete_array<double>(work, lwork);
  Tucker::MemoryManager::safe_delete_array<double>(T, TSize);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(YTransposeCopy);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(YTranspose);
  Tucker::MemoryManager::safe_delete<Tucker::Matrix>(Y);

  std::cout << "Explicity transpose takes: " << transposeTimer.duration() << " seconds. \n";
  //std::cout << "work space query takes: " << qrfWorkSpaceQuerryTimer.duration() << " seconds. \n";
  std::cout << "dgeqrf takes: " << avgQrfTime << " seconds. \n";
  std::cout << "dgeqrt takes: " << avgQrtTime << " seconds. \n";
  std::cout << "dgeqr takes: " << avgQrTime << " seconds. \n";
  //std::cout << "lq work space query takes: " << lqfWorkSpaceQueryTimer.duration() << " seconds. \n";
  std::cout << "dgelqf takes: " << avgLqfTime << " seconds. \n";
  std::cout << "dgelqt takes: " << avgLqtTime << " seconds, \n";
  std::cout << "dgelq takes: " << avgLqTime << " seconds, \n";
  return EXIT_SUCCESS;
}