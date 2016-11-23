#include "Tucker.hpp"
#include<limits>

int main(int argc, char* argv[])
{
  // Read a tensor from a text file
  std::string filename = "input_files/3x5x7x11.txt";
  Tucker::Tensor* Y = Tucker::importTensor(filename.c_str());

  // Write the tensor to a binary file
  filename = "output_files/output.mpi";
  Tucker::exportTensorBinary(Y, filename.c_str());

  // Read a tensor from the binary file
  Tucker::Tensor Y2(Y->size());
  Tucker::importTensorBinary(filename.c_str(), &Y2);

  if(!isApproxEqual(Y, &Y2, 1e-10)) {
    std::cout << "Y and Y2 are not equal\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
