#ifndef TUCKERMPI_HPP_
#define TUCKERMPI_HPP_

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
#include "Tucker_TuckerTensor.hpp"
#include "TuckerOnNode_MetricData.hpp"
#include "Tucker_deep_copy.hpp"
#include "Tucker_create_mirror.hpp"
#include "TuckerMpi_Tensor_io.hpp"
#include "TuckerMpi_compute_slice_metrics.hpp"
#include "TuckerMpi_write_statistics.hpp"
#include "TuckerMpi_normalize_tensor.hpp"

// finally functions related/needed by sthosvd
#include "TuckerMpi_compute_gram.hpp"
#include "TuckerMpi_ttm.hpp"
#include "Tucker_create_core_tensor_truncator.hpp"
#include "TuckerMpi_sthosvd.hpp"

#endif  // TUCKERMPI_HPP_
