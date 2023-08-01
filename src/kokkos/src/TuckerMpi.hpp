#ifndef TUCKER_KOKKOS_MPI_SINGLE_INCLUDE_HPP_
#define TUCKER_KOKKOS_MPI_SINGLE_INCLUDE_HPP_

// NOTE that below the order of the includes is intentional
// and based on the actualy dependencies

// first include headers that only depende on TPLs
#include "Tucker_boilerplate_view_io.hpp"

// then include cmake-based config
#include "Tucker_cmake_config.h"

// then actualy library headers, starting from
// fwd decl and tensor class and operations on it
#include "Tucker_fwd.hpp"
#include "TuckerMpi_Map.hpp"
#include "TuckerMpi_ProcessorGrid.hpp"
#include "TuckerMpi_Distribution.hpp"
#include "TuckerMpi_Tensor.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include "TuckerMpi_Tensor_io.hpp"

// finally functions related/needed by sthosvd
#include "TuckerMpi_ttm.hpp"
#include "Tucker_create_core_tensor_truncator.hpp"
#include "TuckerMpi_sthosvd.hpp"

#endif
