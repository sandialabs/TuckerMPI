
set(TUCKER_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(KokkosKernels REQUIRED)
set(KOKKOS_TPLS
  Kokkos::BLAS Kokkos::LAPACK Kokkos::kokkos Kokkos::kokkoskernels)

#
# set some options
#
option(TUCKER_ENABLE_DEBUG_PRINTS OFF)
option(TUCKER_ENABLE_FALLBACK_VIA_HOST OFF)
if(TUCKER_ENABLE_FALLBACK_VIA_HOST)
  MESSAGE(STATUS "TUCKER_ENABLE_FALLBACK_VIA_HOST=${TUCKER_ENABLE_FALLBACK_VIA_HOST}, building code executing via host\n")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/tucker_cmake_config.h.in tucker_cmake_config.h @ONLY)

#
# branch if onnode or mpi
#
if(TUCKER_IMPL_ENABLE_KOKKOSONLY)
  MESSAGE(STATUS "TUCKER_ENABLE_KOKKOS=${TUCKER_ENABLE_KOKKOS}, building code with Kokkos-only\n")

  set(ALL_TPLS_NEEDED ${KOKKOS_TPLS})

  set(KokkosOnlyTuckerLib kokkosonly_tucker)
  set(SOURCES)
  LIST(APPEND SOURCES
    ${TUCKER_SRCDIR}/impl/Tucker_BlasWrapper.cpp
  )
  ADD_LIBRARY(${KokkosOnlyTuckerLib} SHARED ${SOURCES})
  TARGET_LINK_LIBRARIES(${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED})

  include_directories(${CMAKE_CURRENT_BINARY_DIR} ${TUCKER_SRCDIR} ${TUCKER_SRCDIR}/impl)

  set(STHOSVD_DRIVER_NAME KokkosTucker_sthosvd)
  set(STHOSVD_DRIVER_SRCS)
  LIST(APPEND STHOSVD_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/onnode/TuckerOnNode_sthosvd.cpp
  )
  add_executable(${STHOSVD_DRIVER_NAME} ${STHOSVD_DRIVER_SRCS})
  target_include_directories(${STHOSVD_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${STHOSVD_DRIVER_NAME} ${KokkosOnlyTuckerLib} ${KOKKOS_TPLS})

  add_subdirectory(unit_tests)
  add_subdirectory(drivers_tests)
endif()

if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  MESSAGE(STATUS "TUCKER_ENABLE_KOKKOS=${TUCKER_ENABLE_KOKKOS} and TUCKER_ENABLE_MPI=${TUCKER_ENABLE_MPI}, building code with MPI+Kokkos\n")

  find_package(MPI REQUIRED)

  set(ALL_TPLS_NEEDED ${KOKKOS_TPLS} ${MPI_CXX_LIBRARIES})

  set(MpiKokkosTuckerLib mpikokkos_tucker)
  set(SOURCES)
  LIST(APPEND SOURCES
    ${TUCKER_SRCDIR}/impl/Tucker_BlasWrapper.cpp
    ${TUCKER_SRCDIR}/impl/TuckerMpi_MPIWrapper.cpp
    ${TUCKER_SRCDIR}/TuckerMpi_Map.cpp
    ${TUCKER_SRCDIR}/TuckerMpi_ProcessorGrid.cpp
    ${TUCKER_SRCDIR}/TuckerMpi_Distribution.cpp
  )
  ADD_LIBRARY(${MpiKokkosTuckerLib} SHARED ${SOURCES})
  TARGET_LINK_LIBRARIES(${MpiKokkosTuckerLib} ${ALL_TPLS_NEEDED})

  include_directories(${MPI_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${TUCKER_SRCDIR} ${TUCKER_SRCDIR}/impl)

  set(STHOSVD_DRIVER_NAME MpiKokkosTucker_sthosvd)
  set(STHOSVD_DRIVER_SRCS)
  LIST(APPEND STHOSVD_DRIVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/CmdLineParse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/ParameterFileParserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/mpi/TuckerMpi_sthosvd.cpp
  )
  add_executable(${STHOSVD_DRIVER_NAME} ${STHOSVD_DRIVER_SRCS})
  target_include_directories(${STHOSVD_DRIVER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drivers)
  target_link_libraries(${STHOSVD_DRIVER_NAME} ${MpiKokkosTuckerLib} ${ALL_TPLS_NEEDED})

  add_subdirectory(unit_tests)
  add_subdirectory(drivers_tests)
endif()
