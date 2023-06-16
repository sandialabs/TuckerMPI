set(SHARED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
set(ONNODE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../on-node)

find_package(MPI)
if(NOT ${MPI_FOUND})
  message(FATAL_ERROR "mpi-kokkos requires MPI, library not found")
endif()

set(MpiKokkosTuckerLib mpikokkos_tucker)
set(SOURCES)
LIST(APPEND SOURCES
  ${SHARED_SRC}/Tucker_CmdLineParse.cpp
  ${SHARED_SRC}/Tucker_ParameterFileParserUtils.cpp
  ${SHARED_SRC}/Tucker_BlasWrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TuckerMPI_MPIWrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TuckerMpi_Map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TuckerMpi_ProcessorGrid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TuckerMpi_Distribution.cpp
)
ADD_LIBRARY(${MpiKokkosTuckerLib} SHARED ${SOURCES})

find_package(KokkosKernels REQUIRED)
set(ALL_TPLS_NEEDED Kokkos::kokkos Kokkos::kokkoskernels ${MPI_CXX_LIBRARIES})
TARGET_LINK_LIBRARIES(${MpiKokkosTuckerLib} ${ALL_TPLS_NEEDED})

include_directories(${SHARED_SRC} ${ONNODE_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_CXX_INCLUDE_PATH})

# add tests
add_subdirectory(tests)

#
# create executable for sthosvd
#
set(STHOSVD_SOURCES)
LIST(APPEND STHOSVD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/TuckerMpi_sthosvd.cpp
)
add_executable(MpiKokkosTucker_sthosvd ${STHOSVD_SOURCES})
target_link_libraries(MpiKokkosTucker_sthosvd ${MpiKokkosTuckerLib} ${ALL_TPLS_NEEDED})
