#include "Tucker.hpp"
#include<limits>

int main(int argc, char* argv[])
{
  typedef double scalar_t; // specify precision

  // Read a tensor from a text file
  std::string filename = "input_files/3x5x7x11.txt";
  Tucker::Tensor<scalar_t>* Y = Tucker::importTensor<scalar_t>(filename.c_str());

  // Write the tensor to a binary file
  filename = "output_files/output.mpi";
  Tucker::exportTensorBinary(Y, filename.c_str());

  // Read a tensor from the binary file
  Tucker::Tensor<scalar_t>* Y2 =
      Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(Y->size());
  Tucker::importTensorBinary(Y2, filename.c_str());

  if(!Tucker::isApproxEqual(Y, Y2, 100 * std::numeric_limits<scalar_t>::epsilon())) {
    std::cout << "Y and Y2 are not equal\n";
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(Y);
  Tucker::MemoryManager::safe_delete(Y2);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
