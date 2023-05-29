
find_package(KokkosKernels REQUIRED HINTS ${KOKKOSKERNELS_PFX})

include_directories(${MPI_CXX_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(mainExe ${CMAKE_CURRENT_SOURCE_DIR}/TuckerMpiKokkos_sthosvd.cpp)
target_link_libraries(mainExe mpi_kokkos_tucker Kokkos::kokkos Kokkos::kokkoskernels ${MPI_CXX_LIBRARIES})
