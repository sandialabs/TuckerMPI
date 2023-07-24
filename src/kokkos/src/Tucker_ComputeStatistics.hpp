#ifndef TUCKER_KOKKOS_COMPUTE_STATISTICS_HPP_
#define TUCKER_KOKKOS_COMPUTE_STATISTICS_HPP_

#include <fstream>

namespace Tucker{

template <class TensorType, class ScalarType>
auto compute_statistics(TensorType dataTensor, int scaleMode,
                std::string statsFile, ScalarType stdThresh)
{

    std::cout << "yes" << std::endl;

}

}//end namespace Tucker

#endif
