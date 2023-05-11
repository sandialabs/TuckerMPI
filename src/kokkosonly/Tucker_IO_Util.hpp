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
 * \brief Contains I/O functions for %Tucker code
 *
 * \author Woody Austin
 * \author Grey Ballard
 * \author Alicia Klinvex
 */

#ifndef TUCKER_IO_UTIL_HPP_
#define TUCKER_IO_UTIL_HPP_

#include <string>
#include <vector>
#include <iostream>
#include <limits>
#include <fstream>
#include "Tucker_SizeArray.hpp"
#include "Tucker_Tensor.hpp"
#include "Tucker_TuckerTensor.hpp"

namespace TuckerKokkos{

/** \brief Parses a single potential argument
 *
 * \param[in] argc Number of command line arguments
 * \param[in] argv Array of command line arguments
 * \param[in] cl_arg Command line argument to search for
 * \param[in] default_value Default value of the argument
 *
 * Example usage: \n
 * <tt>
 * std::string paramfn = Tucker::parseString(argc, argv,
 *    "--parameter-file", "paramfile.txt"); \n
 * </tt>
 * This function determines whether <tt>--parameter-file</tt>
 * was a command line argument.  If so, copy its value to
 * <tt>paramfn</tt>.  Otherwise, set <tt>paramfn</tt> equal
 * to <tt>paramfile.txt</tt>.
 */
std::string parseString(const int argc, const char* argv[],
    const std::string& cl_arg, const std::string& default_value);

/** \brief Parses a file into a vector of strings
 *
 * \param[in] paramfn Name of parameter file to be opened
 */
std::vector<std::string> getFileAsStrings(const std::string& paramfn);

/** \brief Parses a single option
 *
 * \param[in] lines Vector of strings; each string represents a single
 * option defined by the user
 * \param[in] keyword Option to be parsed
 * \param[in] default_value Default value of the option
 *
 * \note This templated function has to be defined in the header file
 */
template<typename T>
T stringParse(const std::vector<std::string>& lines,
	      const std::string& keyword,
	      const T& default_value)
{
  T value = default_value;
  for (auto line : lines) {
    // If the keyword is in the string then use that value
    if (line.find(keyword) != std::string::npos) {
      // Find the equal sign
      std::size_t equalPos = line.find("=");
      // Extract the string after that equal sign
      std::string valueSubstring = line.substr(equalPos+1);

      // This is explicitly for bool arguments:
      // In both, the second clause makes sure that filenames with "true" or "false" in them
      // are not replaced by 0 or 1
      if (valueSubstring.find("true") != std::string::npos &&
          valueSubstring.find_first_not_of("true \t") == std::string::npos) {

        valueSubstring = "1";
      } else if (valueSubstring.find("false") != std::string::npos &&
          valueSubstring.find_first_not_of("true \t") == std::string::npos) {

        valueSubstring = "0";
      }

      std::stringstream valueStream(valueSubstring);
      // The value should be one "word", extract it from the string
      valueStream >> value;
      break;
    }
  }

  return value;
}

/** \brief Parse SizeArray from vector of lines
 *
 * \param[in] lines Vector of strings; each string represents a single
 * option defined by the user
 * \param[in] keyword Option to be parsed
 *
 * \note User is responsible for deallocating the SizeArray
 */
SizeArray stringParseSizeArray(const std::vector<std::string>& lines,
				       const std::string& keyword);

template <class ScalarType, class MemorySpace>
void importTensorBinary(Tensor<ScalarType, MemorySpace> & X,
			const char* filename)
{
  // Get the maximum file size we can read
  const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  std::ifstream ifs;
  ifs.open(filename, std::ios::in | std::ios::binary);
  assert(ifs.is_open());

  std::streampos begin, end, size;
  begin = ifs.tellg();
  ifs.seekg(0, std::ios::end);
  end = ifs.tellg();
  size = end - begin;
  //std::cout << "Reading " << size << " bytes...\n";
  size_t numEntries = X.getNumElements();
  assert(size == numEntries*sizeof(ScalarType));

  // Read the file
  auto view1d_d = X.data();
  auto view1d_h = Kokkos::create_mirror(view1d_d);
  ScalarType* data = view1d_h.data();
  ifs.seekg(0, std::ios::beg);
  ifs.read((char*)data,size);

  Kokkos::deep_copy(view1d_d, view1d_h);
  ifs.close();
}

template <class ScalarType, class MemorySpace>
void readTensorBinary(Tensor<ScalarType, MemorySpace> & Y,
		      const char* filename)
{
  std::ifstream inStream(filename);
  std::string temp;
  int nfiles = 0;
  while(inStream >> temp) { nfiles++; }
  inStream.close();
  if(nfiles != 1) {
    throw std::runtime_error("readTensorBinary hardwired for one file only for now");
  }
  importTensorBinary(Y, temp.c_str());
}

template <class ScalarType, class MemorySpace>
void printEigenvalues(const TuckerTensor<ScalarType, MemorySpace> & factorization,
		      const std::string& filePrefix,
		      bool useLQ)
{
  const int nmodes = factorization.numDims();

  for(int mode=0; mode<nmodes; mode++) {
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";
    std::ofstream ofs(ss.str());
    // Determine the number of eigenvalues for this mode
    auto eigVals_view = factorization.eigValsAt(mode);
    auto eigVals_view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
							      eigVals_view);
    const int nevals = eigVals_view.extent(0);

    // if(useLQ){
    //   for(int i=0; i<nevals; i++) {
    //     ofs << std::setprecision(16)
    // 	    << std::pow(factorization->singularValues[mode][i], 2)
    // 	    << std::endl;
    //   }
    // }
    // else{
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16)
	    << eigVals_view_h(i)
	    << std::endl;
      }
   //}
    ofs.close();
  }
}

template <class scalar_t, class mem_space>
void exportTensorBinary(const Tensor<scalar_t, mem_space> & Y, const char* filename)
{
  const std::streamoff MAX_OFFSET = std::numeric_limits<std::streamoff>::max();
  size_t numEntries = Y.getNumElements();
  // Open file
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::binary);
  assert(ofs.is_open());
  const scalar_t* data = Y.data().data();
  ofs.write((char*)data,numEntries*sizeof(scalar_t));
  ofs.close();
}

template <class ScalarType, class mem_space>
Tensor<ScalarType, mem_space> importTensor(const char* filename)
{
  // Open file
  std::ifstream ifs;
  ifs.open(filename);
  if (ifs.is_open()) {
    std::cout<< "file open, it's working!\n";
  }else{
    std::cout<< "fail to open file...\n";
  }
  assert(ifs.is_open());

  // Read the type of object
  // If the type is not "tensor", that's bad
  std::string tensorStr;

  ifs >> tensorStr;
  std::cout << "tensorStr: " << tensorStr << "\n";
  assert(tensorStr == "tensor" || tensorStr == "matrix");

  // Read the number of dimensions
  int ndims;
  ifs >> ndims;
  std::cout << "ndims: " << ndims << "\n";

  // Create a SizeArray of that length
  SizeArray sz(ndims);

  // Read the dimensions
  for(int i=0; i<ndims; i++) {
    ifs >> sz[i];
    //std::cout << "sz[i]: " << sz[i] << "\n";
  }

  // Create a tensor using that SizeArray
  Tensor<ScalarType, mem_space> t(sz);
  // std::cout << "print: \n";
  // t.print();

  // Read the entries of the tensor
  // TODO
  
  // Close the file
  ifs.close();

  // Return the tensor
  return t;
}

}// end namespace TuckerKokkos

#endif /* TUCKER_IO_UTIL_HPP_ */
