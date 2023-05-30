set(MpiKokkosTuckerLib mpi_kokkos_tucker)
set(SOURCES)
LIST(APPEND SOURCES 
  MpiKokkos_Tucker_SizeArray.cpp 
)
ADD_LIBRARY(${MpiKokkosTuckerLib} SHARED ${SOURCES})

find_package(KokkosKernels REQUIRED HINTS ${KOKKOSKERNELS_PFX})
set(ALL_TPLS_NEEDED
  Kokkos::kokkos Kokkos::kokkoskernels)
TARGET_LINK_LIBRARIES(${MpiKokkosTuckerLib} ${ALL_TPLS_NEEDED})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${MPI_CXX_INCLUDE_PATH})
# include_directories(../kokkosonly)

set(STHOSVD_SOURCES)
LIST(APPEND STHOSVD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/MpiKokkos_Tucker_CmdLineParse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/MpiKokkos_Tucker_sthosvd.cpp
)

add_executable(mpikokkos_tucker ${STHOSVD_SOURCES})

target_link_libraries(mpikokkos_tucker ${MpiKokkosTuckerLib} ${ALL_TPLS_NEEDED} ${MPI_CXX_LIBRARIES})
