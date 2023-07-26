#ifndef TUCKER_PRINT_TUCKER_EIGENVALUES_HPP_
#define TUCKER_PRINT_TUCKER_EIGENVALUES_HPP_

#include <Kokkos_Core.hpp>
#include <iomanip>
#include <fstream>

// FIXME: need to figure out
// https://gitlab.com/nga-tucker/TuckerMPI/-/issues/15
namespace Tucker{

namespace impl{
struct ReturnSelf{
  template<class T> T operator()(const T & v) const { return v; }
};
}

template <class StorageType, class UnaryOp = impl::ReturnSelf>
void print_eigenvalues(StorageType container,
		       std::string filePrefix,
		       UnaryOp op = UnaryOp())
{

  const int nmodes = container.rank();
  for(int mode=0; mode<nmodes; mode++)
  {
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";
    // Open the file
    std::ofstream ofs(ss.str());
    std::cout << "Writing singular values to " << ss.str() << std::endl;

    // Determine the number of eigenvalues for this mode
    auto eigvals = container.eigenvalues(mode);
    auto eigvals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigvals);
    const int nevals = eigvals.extent(0);

    for(int i=0; i<nevals; i++) {
      ofs << std::setprecision(16) << op(eigvals_h(i)) << std::endl;
    }
    ofs.close();
  }
}

}
#endif
