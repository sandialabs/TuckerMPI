#include<cassert>
#include<cstdlib>
#include<cmath>
#include "Tucker_IO_Util.hpp"
#include "Tucker.hpp"
#include "Tucker_BlasWrapper.hpp"

int main(int argc, char* argv[])
{

// specify precision
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif

  //int a = mkl_get_num_threads();
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
  Tucker::Timer syrkTimer;

  Tucker::Matrix<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(YNrows, YNcols);
  int sizeOfY = YNrows*YNcols;
  Y->rand();
  Tucker::Matrix<scalar_t>* YCopy = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(YNrows, YNcols);

  int info;
  //this tau is for both dgeqrf and dgelqf.
  scalar_t* tau = Tucker::MemoryManager::safe_new_array<scalar_t>(std::min(YNrows, YNcols));
  scalar_t* work = Tucker::MemoryManager::safe_new_array<scalar_t>(1);
  scalar_t* T;
  int lwork;

  //dgeqrf
  //workspace query
  // qrfWorkSpaceQueryTimer.start();
  // Tucker::geqrf(&YNrows, &YNcols, Y->data(), &YNrows, tau, work, &negOne, &info);
  // qrfWorkSpaceQueryTimer.stop();
  // lwork = work[0];
  // Tucker::MemoryManager::safe_delete_array(work, 1);
  // work = Tucker::MemoryManager::safe_new_array<scalar_t>(lwork);
  // //query done
  // for(int i=0; i<avgIteration; i++){
  //   Tucker::copy(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
  //   qrfTimer.start();
  //   Tucker::geqrf(&YNrows, &YNcols, YCopy->data(), &YNrows, tau, work, &lwork, &info);
  //   qrfTimer.stop();
  // }
  // scalar_t avgQrfTime = qrfTimer.duration() / avgIteration;
  // Tucker::MemoryManager::safe_delete_array(work, lwork);
  // std::cout << "qrf lwork: " << lwork << std::endl;
  
  // // //dgeqrt
  // T = Tucker::MemoryManager::safe_new_array<scalar_t>(nb*YNcols);
  // work = Tucker::MemoryManager::safe_new_array<scalar_t>(nb*YNcols);
  // for(int i=0; i<avgIteration; i++){
  //   Tucker::copy(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
  //   qrtTimer.start();
  //   Tucker::geqrt(&YNrows, &YNcols, &nb, YCopy->data(), &YNrows, T, &nb, work, &info);
  //   qrtTimer.stop();
  // }
  // scalar_t avgQrtTime = qrtTimer.duration() /avgIteration;
  // Tucker::MemoryManager::safe_delete_array(work, nb*YNcols);
  // Tucker::MemoryManager::safe_delete_array(T, nb*YNcols);

  // //dgeqr
  qrWorkSpaceQueryTimer.start();
  work = Tucker::MemoryManager::safe_new_array<scalar_t>(1);
  //T has to have space for at least 5 doubles, see dgeqr.
  T = Tucker::MemoryManager::safe_new_array<scalar_t>(5);
  Tucker::geqr(&YNrows, &YNcols, Y->data(), &YNrows, T, &negOne, work, &negOne, &info);
  lwork = work[0];
  int TSize = T[0];
  Tucker::MemoryManager::safe_delete_array(work, 1);
  Tucker::MemoryManager::safe_delete_array(T, 5);
  qrWorkSpaceQueryTimer.stop();
  std::cout << "qr lwork:" << lwork << " TSize: " << TSize << std::endl;
  work = Tucker::MemoryManager::safe_new_array<scalar_t>(lwork);
  T = Tucker::MemoryManager::safe_new_array<scalar_t>(TSize);
  for(int i=0; i<avgIteration; i++){
    Tucker::copy(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
    qrTimer.start();
    Tucker::geqr(&YNrows, &YNcols, YCopy->data(), &YNrows, T, &TSize, work, &lwork, &info);
    qrTimer.stop();
  }
  scalar_t avgQrTime = qrTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete_array(work, lwork);
  Tucker::MemoryManager::safe_delete_array(T, TSize);

  //dsyrk
  scalar_t* gram = Tucker::MemoryManager::safe_new_array<scalar_t>(YNcols*YNcols);
  std::cout << "hello" <<std::endl;
  for(int i=0; i<avgIteration; i++){
    Tucker::copy(&sizeOfY, Y->data(), &one, YCopy->data(), &one);
    std::cout << "hello" <<std::endl;
    syrkTimer.start();
    char uplo = 'U';
    char trans = 'T';
    scalar_t alpha = 1;
    scalar_t beta = 0;
    Tucker::syrk(&uplo, &trans, &YNcols, &YNrows, &alpha,
        YCopy->data(), &YNrows, &beta, gram, &YNcols);
    syrkTimer.stop();
  }
  scalar_t avgsyrkTime = syrkTimer.duration() / avgIteration;
   Tucker::MemoryManager::safe_delete(YCopy);
  Tucker::MemoryManager::safe_delete_array(gram, YNcols*YNcols);

  //Get YTranspose
  transposeTimer.start();
  Tucker::Matrix<scalar_t>* YTranspose = Y->getTranspose();
  transposeTimer.stop();
  Tucker::Matrix<scalar_t>* YTransposeCopy = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(YTranspose->nrows(), YTranspose->ncols());

  // //dgelqf
  // //workspace query
  // lqfWorkSpaceQueryTimer.start();
  // work = Tucker::MemoryManager::safe_new_array<scalar_t>(1);
  // Tucker::gelqf(&YNcols, &YNrows, YTranspose->data(), &YNcols, tau, work, &negOne, &info);
  // lwork = work[0];
  // lqfWorkSpaceQueryTimer.stop();
  // std::cout << "lqf lwork: " << lwork << std::endl;
  // Tucker::MemoryManager::safe_delete_array(work, 1);
  // work = Tucker::MemoryManager::safe_new_array<scalar_t>(lwork);
  // //query done
  // for(int i=0; i<avgIteration; i++){
  //   Tucker::copy(&sizeOfY, YTranspose->data(), &one, YTransposeCopy->data(), &one);
  //   lqfTimer.start();
  //   Tucker::gelqf(&YNcols, &YNrows, YTransposeCopy->data(), &YNcols, tau, work, &lwork, &info);
  //   lqfTimer.stop();
  // }
  // std::cout << "lqf done" << std::endl;
  // scalar_t avgLqfTime = lqfTimer.duration() / avgIteration;
  // Tucker::MemoryManager::safe_delete_array(work, lwork);  
  // Tucker::MemoryManager::safe_delete_array(tau, std::min(YNrows, YNcols));

  // //dgelqt
  // T = Tucker::MemoryManager::safe_new_array<scalar_t>(nb*YNrows);
  // work = Tucker::MemoryManager::safe_new_array<scalar_t>(nb*YNcols);
  // for(int i=0; i<avgIteration; i++){
  //   Tucker::copy(&sizeOfY, YTranspose->data(), &one, YTransposeCopy->data(), &one);
  //   lqtTimer.start();
  //   Tucker::gelqt(&YNcols, &YNrows, &nb, YTransposeCopy->data(), &YNcols, T, &nb, work, &info);
  //   lqtTimer.stop();
  // }
  // scalar_t avgLqtTime = lqtTimer.duration() / avgIteration;
  // std::cout << "lqt done." << std::endl;
  // Tucker::MemoryManager::safe_delete_array(T, nb*YNrows);
  // Tucker::MemoryManager::safe_delete_array(work, nb*YNrows);

  //dgelq
  lqWorkSpaceQueryTimer.start();
  work = Tucker::MemoryManager::safe_new_array<scalar_t>(1);
  T = Tucker::MemoryManager::safe_new_array<scalar_t>(5);
  Tucker::gelq(&YNcols, &YNrows, YTranspose->data(), &YNcols, T, &negOne, work, &negOne, &info);
  lwork = work[0];
  TSize = T[0];
  Tucker::MemoryManager::safe_delete_array(work, 1);
  Tucker::MemoryManager::safe_delete_array(T, 5);
  lqWorkSpaceQueryTimer.stop();
  std::cout << "lq lwork:" << lwork << " TSize: " << TSize << std::endl;
  work = Tucker::MemoryManager::safe_new_array<scalar_t>(lwork);
  T = Tucker::MemoryManager::safe_new_array<scalar_t>(TSize);
  for(int i=0; i<avgIteration; i++){
    Tucker::copy(&sizeOfY, YTranspose->data(), &one, YTransposeCopy->data(), &one);
    lqTimer.start();
    Tucker::gelq(&YNcols, &YNrows, YTransposeCopy->data(), &YNcols, T, &TSize, work, &lwork, &info);
    lqTimer.stop();
  }
  scalar_t avgLqTime = lqTimer.duration() / avgIteration;
  Tucker::MemoryManager::safe_delete_array(work, lwork);
  Tucker::MemoryManager::safe_delete_array(T, TSize);
  Tucker::MemoryManager::safe_delete(YTransposeCopy);
  Tucker::MemoryManager::safe_delete(YTranspose);
  Tucker::MemoryManager::safe_delete(Y);

  std::cout << "Explicity transpose takes: " << transposeTimer.duration() << " seconds. \n";
  std::cout << "work space query takes: " << qrfWorkSpaceQueryTimer.duration() << " seconds. \n";
  // std::cout << "geqrf takes: " << avgQrfTime << " seconds. \n";
  // std::cout << "geqrt takes: " << avgQrtTime << " seconds. \n";
  std::cout << "geqr takes: " << avgQrTime << " seconds. \n";
  std::cout << "syrk takes: " << avgsyrkTime << " seconds. \n";
  std::cout << "lq work space query takes: " << lqfWorkSpaceQueryTimer.duration() << " seconds. \n";
  // std::cout << "gelqf takes: " << avgLqfTime << " seconds. \n";
  // std::cout << "gelqt takes: " << avgLqtTime << " seconds, \n";
  std::cout << "gelq takes: " << avgLqTime << " seconds, \n";
  return EXIT_SUCCESS;
}