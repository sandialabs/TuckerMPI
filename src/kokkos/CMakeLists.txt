
set(TUCKER_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(KokkosKernels REQUIRED)
set(KOKKOS_TPLS Kokkos::kokkoskernels) # only need this to transitively access core, blas, etc

set(TUCKER_ALL_TPLS_NEEDED ${KOKKOS_TPLS})


# if CUBLAS is ON, we need cusolver too because Kernels does not pick it up
if (TARGET Kokkos::CUBLAS)
if (NOT TARGET CUDA::cusolver)
  find_package(CUDAToolkit)
  message(STATUS "CUDA::cusolver not found, finding CUDAToolkit")
  list(APPEND TUCKER_ALL_TPLS_NEEDED CUDA::cusolver)
endif()
endif()

# if ROCBLAS is ON, we need rocsolver too because Kernels does not pick it up
if (Kokkos_ENABLE_HIP)
if (NOT TARGET roc::rocblas)
  find_package(ROCBLAS)
  message(STATUS "roc::rocblas not found, finding ROCBLAS")
  list(APPEND TUCKER_ALL_TPLS_NEEDED roc::rocblas)
endif()
if (NOT TARGET roc::rocsolver)
  find_package(ROCSOLVER)
  message(STATUS "roc::rocsolver not found, finding ROCSOLVER")
  list(APPEND TUCKER_ALL_TPLS_NEEDED roc::rocsolver)
endif()
endif()

#
# set some options
#
option(TUCKER_ENABLE_MPI OFF)
option(TUCKER_ENABLE_DEBUG_PRINTS OFF)
option(TUCKER_ENABLE_FALLBACK_VIA_HOST OFF)
if(TUCKER_ENABLE_FALLBACK_VIA_HOST)
  MESSAGE(STATUS "TUCKER_ENABLE_FALLBACK_VIA_HOST=${TUCKER_ENABLE_FALLBACK_VIA_HOST}, building code executing via host\n")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/Tucker_cmake_config.h.in Tucker_cmake_config.h @ONLY)

# =========================================
# handle onnode only case
# =========================================
if(TUCKER_IMPL_ENABLE_KOKKOSONLY)
  MESSAGE(STATUS "TUCKER_ENABLE_KOKKOS=${TUCKER_ENABLE_KOKKOS}, building code with Kokkos-only\n")

  set(KokkosOnlyTuckerLib kokkosonly_tucker)
  set(ONNODE_SOURCES)
  LIST(APPEND ONNODE_SOURCES ${TUCKER_SRCDIR}/impl/Tucker_BlasWrapper.cpp)
  LIST(APPEND ONNODE_SOURCES ${TUCKER_SRCDIR}/TuckerOnNode_ISVD.cpp)
  LIST(APPEND ONNODE_SOURCES ${TUCKER_SRCDIR}/TuckerOnNode_StreamingTuckerTensor.cpp)
  ADD_LIBRARY(${KokkosOnlyTuckerLib} SHARED ${ONNODE_SOURCES})
  TARGET_LINK_LIBRARIES(${KokkosOnlyTuckerLib} ${TUCKER_ALL_TPLS_NEEDED})
  include_directories(${CMAKE_CURRENT_BINARY_DIR} ${TUCKER_SRCDIR})

  #
  # driver KokkosTucker_sthosvd
  #
  set(ONNODE_STHOSVD_DRIVER_NAME KokkosTucker_sthosvd)
  set(ONNODE_STHOSVD_DRIVER_SRCS)
  LIST(APPEND ONNODE_STHOSVD_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/onnode/TuckerOnNode_sthosvd.cpp
  )
  add_executable(${ONNODE_STHOSVD_DRIVER_NAME} ${ONNODE_STHOSVD_DRIVER_SRCS})
  target_include_directories(${ONNODE_STHOSVD_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${ONNODE_STHOSVD_DRIVER_NAME} ${KokkosOnlyTuckerLib} ${KOKKOS_TPLS})

  #
  # driver KokkosTucker_generate
  #
  set(ONNODE_GENERATE_DRIVER_NAME KokkosTucker_generate)
  set(ONNODE_GENERATE_DRIVER_SRCS)
  LIST(APPEND ONNODE_GENERATE_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/onnode/TuckerOnNode_generate.cpp
  )
  add_executable(${ONNODE_GENERATE_DRIVER_NAME} ${ONNODE_GENERATE_DRIVER_SRCS})
  target_include_directories(${ONNODE_GENERATE_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${ONNODE_GENERATE_DRIVER_NAME} ${KokkosOnlyTuckerLib} ${KOKKOS_TPLS})

  #
  # driver KokkosTucker_reconstruct
  #
  set(ONNODE_RECONSTRUCT_DRIVER_NAME KokkosTucker_reconstruct)
  set(ONNODE_RECONSTRUCT_DRIVER_SRCS)
  LIST(APPEND ONNODE_RECONSTRUCT_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/onnode/TuckerOnNode_reconstruct.cpp
  )
  add_executable(${ONNODE_RECONSTRUCT_DRIVER_NAME} ${ONNODE_RECONSTRUCT_DRIVER_SRCS})
  target_include_directories(${ONNODE_RECONSTRUCT_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${ONNODE_RECONSTRUCT_DRIVER_NAME} ${KokkosOnlyTuckerLib} ${KOKKOS_TPLS})

  #
  # driver KokkosTucker_streaming_sthosvd
  #
  set(ONNODE_STREAMING_STHOSVD_DRIVER_NAME KokkosTucker_streaming_sthosvd)
  set(ONNODE_STREAMING_STHOSVD_DRIVER_SRCS)
  LIST(APPEND ONNODE_STREAMING_STHOSVD_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/onnode/TuckerOnNode_streaming_sthosvd.cpp
  )
  add_executable(${ONNODE_STREAMING_STHOSVD_DRIVER_NAME} ${ONNODE_STREAMING_STHOSVD_DRIVER_SRCS})
  target_include_directories(${ONNODE_STREAMING_STHOSVD_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${ONNODE_STREAMING_STHOSVD_DRIVER_NAME} ${KokkosOnlyTuckerLib} ${KOKKOS_TPLS})

  #
  # add tests
  #
  add_subdirectory(unit_tests)
  add_subdirectory(drivers_tests)
endif()


# =========================================
# handle MPI case
# =========================================
if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  MESSAGE(STATUS "TUCKER_ENABLE_KOKKOS=${TUCKER_ENABLE_KOKKOS} and TUCKER_ENABLE_MPI=${TUCKER_ENABLE_MPI}, building code with MPI+Kokkos\n")

  set(TUCKER_ENABLE_MPI ON)
  find_package(MPI REQUIRED)
  list(APPEND TUCKER_ALL_TPLS_NEEDED ${MPI_CXX_LIBRARIES})

  set(MpiKokkosTuckerLib mpikokkos_tucker)
  set(MPI_SOURCES)
  LIST(APPEND MPI_SOURCES
    ${TUCKER_SRCDIR}/impl/Tucker_BlasWrapper.cpp
    ${TUCKER_SRCDIR}/impl/TuckerMpi_MPIWrapper.cpp
    ${TUCKER_SRCDIR}/TuckerMpi_Map.cpp
    ${TUCKER_SRCDIR}/TuckerMpi_ProcessorGrid.cpp
    ${TUCKER_SRCDIR}/TuckerMpi_Distribution.cpp
  )
  ADD_LIBRARY(${MpiKokkosTuckerLib} SHARED ${MPI_SOURCES})
  TARGET_LINK_LIBRARIES(${MpiKokkosTuckerLib} ${TUCKER_ALL_TPLS_NEEDED})
  include_directories(${MPI_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${TUCKER_SRCDIR})

  #
  # driver MpiKokkosTucker_sthosvd
  #
  set(DISTRIBUTED_STHOSVD_DRIVER_NAME MpiKokkosTucker_sthosvd)
  set(DISTRIBUTED_STHOSVD_DRIVER_SRCS)
  LIST(APPEND DISTRIBUTED_STHOSVD_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/mpi/TuckerMpi_sthosvd.cpp
  )
  add_executable(${DISTRIBUTED_STHOSVD_DRIVER_NAME} ${DISTRIBUTED_STHOSVD_DRIVER_SRCS})
  target_include_directories(${DISTRIBUTED_STHOSVD_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${DISTRIBUTED_STHOSVD_DRIVER_NAME} ${MpiKokkosTuckerLib} ${TUCKER_ALL_TPLS_NEEDED})

  #
  # driver MpiKokkosTucker_generate
  #
  set(DISTRIBUTED_GENERATE_DRIVER_NAME MpiKokkosTucker_generate)
  set(DISTRIBUTED_GENERATE_DRIVER_SRCS)
  LIST(APPEND DISTRIBUTED_GENERATE_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/mpi/TuckerMpi_generate.cpp
  )
  add_executable(${DISTRIBUTED_GENERATE_DRIVER_NAME} ${DISTRIBUTED_GENERATE_DRIVER_SRCS})
  target_include_directories(${DISTRIBUTED_GENERATE_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${DISTRIBUTED_GENERATE_DRIVER_NAME} ${MpiKokkosTuckerLib} ${TUCKER_ALL_TPLS_NEEDED})

  add_subdirectory(unit_tests)
  add_subdirectory(drivers_tests)
endif()
