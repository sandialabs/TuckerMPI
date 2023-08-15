
set(TUCKER_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(KokkosKernels REQUIRED)
set(KOKKOS_TPLS Kokkos::kokkoskernels) # only need this to transitively access core, blas, etc

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

#
# branch if onnode or mpi
#
if(TUCKER_IMPL_ENABLE_KOKKOSONLY)
  MESSAGE(STATUS "TUCKER_ENABLE_KOKKOS=${TUCKER_ENABLE_KOKKOS}, building code with Kokkos-only\n")

  set(TUCKER_ALL_TPLS_NEEDED ${KOKKOS_TPLS})

  set(KokkosOnlyTuckerLib kokkosonly_tucker)
  set(ONNODE_SOURCES)
  LIST(APPEND ONNODE_SOURCES
    ${TUCKER_SRCDIR}/impl/Tucker_BlasWrapper.cpp
  )
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
  # add tests
  #
  add_subdirectory(unit_tests)
  add_subdirectory(drivers_tests)
endif()

if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  MESSAGE(STATUS "TUCKER_ENABLE_KOKKOS=${TUCKER_ENABLE_KOKKOS} and TUCKER_ENABLE_MPI=${TUCKER_ENABLE_MPI}, building code with MPI+Kokkos\n")

  set(TUCKER_ENABLE_MPI ON)
  find_package(MPI REQUIRED)
  set(TUCKER_ALL_TPLS_NEEDED ${KOKKOS_TPLS} ${MPI_CXX_LIBRARIES})

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

  add_subdirectory(unit_tests)
  add_subdirectory(drivers_tests)
endif()
