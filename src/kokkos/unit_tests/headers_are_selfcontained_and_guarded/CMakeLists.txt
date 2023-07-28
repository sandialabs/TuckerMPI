
# NOTE: inspired by the similar test in kokkos-core

# Create tests that contain each header separately.
# The goal is to verify that headers are self-contained and guarded.
# Only *compile* not run.

SET(BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")
file(GLOB TUCKER_COMMON_HEADERS RELATIVE ${BASE_DIR}/src
  ${BASE_DIR}/src/Tucker_*.hpp)
file(GLOB TUCKER_ONNODE_HEADERS RELATIVE ${BASE_DIR}/src
  ${BASE_DIR}/src/TuckerOnNode*.hpp)
if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  file(GLOB TUCKER_MPI_HEADERS RELATIVE ${BASE_DIR}/src
    ${BASE_DIR}/src/TuckerMpi_*.hpp)
endif()

set(TUCKER_HEADERS ${TUCKER_COMMON_HEADERS} ${TUCKER_ONNODE_HEADERS} ${TUCKER_MPI_HEADERS})

message(STATUS ${TUCKER_HEADERS})

foreach (_header ${TUCKER_HEADERS})
  string(REGEX REPLACE "[\./]" "_" header_test_name ${_header})
  set(header_test_name Tucker_HeaderSelfContained_${header_test_name})
  set_source_files_properties(tstHeader.cpp PROPERTIES LANGUAGE CXX)
  add_executable(${header_test_name} tstHeader.cpp)
  target_link_libraries(${header_test_name} PRIVATE ${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED})
  target_compile_definitions(${header_test_name} PRIVATE TUCKER_HEADER_TEST_NAME=${_header})
endforeach()
