#ifndef TUCKER_PRINT_TUCKER_EIGENVALUES_HPP_
#define TUCKER_PRINT_TUCKER_EIGENVALUES_HPP_

#include <Kokkos_Core.hpp>
#include <iomanip>
#include <fstream>

// FIXME: need to figure out
// https://gitlab.com/nga-tucker/TuckerMPI/-/issues/15
namespace Tucker{

template <class TuckerTensorType>
void print_eigenvalues(TuckerTensorType factorization,
		       std::string filePrefix,
		       bool squareBeforeWriting)
{

  const int nmodes = factorization.rank();
  for(int mode=0; mode<nmodes; mode++)
  {
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";
    // Open the file
    std::ofstream ofs(ss.str());
    std::cout << "Writing singular values to " << ss.str() << std::endl;

    // Determine the number of eigenvalues for this mode
    auto eigvals = factorization.eigenvalues(mode);
    const int nevals = eigvals.extent(0);

    if (squareBeforeWriting){
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16) << std::pow(eigvals(i),2) << std::endl;
      }
    }
    else{
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16)
	    << /*sqrt(std::abs(*/eigvals(i)/*))*/ << std::endl;
      }
    }
    ofs.close();
  }
}

}
#endif
