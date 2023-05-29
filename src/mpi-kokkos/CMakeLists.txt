set(SOURCES)
LIST(APPEND SOURCES 
  MpiKokkos_Tucker_SizeArray.cpp
  )
ADD_LIBRARY(mpi_kokkos_tucker SHARED ${SOURCES})

find_package(KokkosKernels REQUIRED HINTS ${KOKKOSKERNELS_PFX})

include_directories(${MPI_CXX_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

set(STHOSVD_SOURCES)
LIST(APPEND STHOSVD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/MpiKokkos_Tucker_CmdLineParse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/MpiKokkos_Tucker_ParameterFileParserUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/MpiKokkos_Tucker_sthosvd.cpp
)

add_executable(mainExe ${STHOSVD_SOURCES})
target_link_libraries(mainExe mpi_kokkos_tucker Kokkos::kokkos Kokkos::kokkoskernels ${MPI_CXX_LIBRARIES})
