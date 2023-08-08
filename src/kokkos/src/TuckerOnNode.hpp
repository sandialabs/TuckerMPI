#ifndef TUCKER_KOKKOS_ONNODE_SINGLE_INCLUDE_HPP_
#define TUCKER_KOKKOS_ONNODE_SINGLE_INCLUDE_HPP_

// NOTE that below the order of the includes is intentional
// and based on the actualy dependencies

// first include headers that only depende on TPLs
#include "Tucker_boilerplate_view_io.hpp"

// then include cmake-based config
#include "Tucker_cmake_config.h"

// then actualy library headers, starting from
// fwd decl and tensor class and operations on it
#include "Tucker_fwd.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include "TuckerOnNode_MetricData.hpp"
#include "Tucker_create_mirror.hpp"
#include "Tucker_deep_copy.hpp"
#include "TuckerOnNode_Tensor_io.hpp"
#include "TuckerOnNode_compute_slice_metrics.hpp"
#include "TuckerOnNode_write_statistics.hpp"
#include "TuckerOnNode_transform_slices.hpp"
#include "TuckerOnNode_normalize_tensor.hpp"

// finally functions related/needed by sthosvd
#include "TuckerOnNode_compute_gram.hpp"
#include "TuckerOnNode_ttm.hpp"
#include "Tucker_create_core_tensor_truncator.hpp"
#include "TuckerOnNode_sthosvd.hpp"

#endif
