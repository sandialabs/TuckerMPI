/** \copyright
 * Copyright (2016) Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 * certain rights in this software.
 * \n\n
 * BSD 2-Clause License
 * \n\n
 * Copyright (c) 2016, Sandia Corporation
 * All rights reserved.
 * \n\n
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * \n\n
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * \n\n
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * .
 * \n
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @file
 * \brief Contains I/O functions for parallel %Tucker code
 *
 * \author Alicia Klinvex
 */

#ifndef TUCKERMPI_IO_UTIL_HPP_
#define TUCKERMPI_IO_UTIL_HPP_

#include<fstream>
#include<sstream>
#include<cmath>

namespace TuckerMPI {

void printSingularValues(const TuckerTensor* factorization,
    std::string filePrefix, bool useLQ)
{
  // For each mode...
  int nmodes = factorization->N;
  for(int mode=0; mode<nmodes; mode++) {
    // Create the filename by appending the mode #
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";

    // Open the file
    std::ofstream ofs(ss.str());
    std::cout << "Writing singular values to " << ss.str() << std::endl;

    // Determine the number of eigenvalues for this mode
    int nevals = factorization->U[mode]->nrows();
    if(useLQ){
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16) << factorization->singularValues[mode][i] << std::endl;
      }
    }
    else{
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16) << sqrt(std::abs(factorization->eigenvalues[mode][i])) << std::endl;
      }
    }
    ofs.close();
  }
}

/**
 * If used in LQ approch, this will print the left singular vector of the L,
 * which should be the same as the eigenvectors in the gram approach up to a
 * sign change.
 */
void printEigenvectors(const TuckerTensor* factorization,
    std::string filePrefix)
{
  // For each mode...
  int nmodes = factorization->N;
  for(int mode=0; mode<nmodes; mode++) {
    // Create the filename by appending the mode #
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";

    // Open the file
    std::ofstream ofs(ss.str());
    std::cout << "Writing eigenvectors to " << ss.str() << std::endl;

    // Determine the number of eigenvalues for this mode
    ofs << factorization->U[mode]->prettyPrint();

    ofs.close();
  }
}

} // end namespace

#endif /* TUCKERMPI_IO_UTIL_HPP_ */
