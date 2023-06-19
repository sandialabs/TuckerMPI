
set(TUCKER_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(KokkosKernels REQUIRED)
set(KOKKOS_TPLS
  Kokkos::BLAS Kokkos::LAPACK Kokkos::kokkos Kokkos::kokkoskernels)

if(TUCKER_ENABLE_KOKKOSONLY)
  MESSAGE(STATUS "\n\nTUCKER_ENABLE_KOKKOSONLY=${TUCKER_ENABLE_KOKKOSONLY}, building code with Kokkos-only\n")

  set(ALL_TPLS_NEEDED ${KOKKOS_TPLS})

  set(KokkosOnlyTuckerLib kokkosonly_tucker)
  set(SOURCES)
  LIST(APPEND SOURCES
    ${TUCKER_SRCDIR}/impl/Tucker_BlasWrapper.cpp
  )
  ADD_LIBRARY(${KokkosOnlyTuckerLib} SHARED ${SOURCES})
  TARGET_LINK_LIBRARIES(${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED})

  include_directories(${TUCKER_SRCDIR} ${TUCKER_SRCDIR}/impl)

  add_subdirectory(tests)

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
endif()

if(TUCKER_ENABLE_MPIKOKKOS)
  MESSAGE(STATUS "\n\nTUCKER_ENABLE_MPIKOKKOS=${TUCKER_ENABLE_MPIKOKKOS}, building code with MPI+Kokkos\n")

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

  include_directories(${TUCKER_SRCDIR} ${TUCKER_SRCDIR}/impl)

  # FIXME: need to add scaffoldign for MPI tests before we add this
  #add_subdirectory(tests)

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

endif()
