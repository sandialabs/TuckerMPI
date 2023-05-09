CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(Tucker_Kokkos)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES)
LIST(APPEND SOURCES
  Tucker_BlasWrapper.cpp
  Tucker_SizeArray.cpp
  Tucker_IO_Util.cpp
  )
ADD_LIBRARY(kokkos_tucker SHARED ${SOURCES})
TARGET_LINK_LIBRARIES(kokkos_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

find_package(KokkosKernels REQUIRED HINTS ${KOKKOSKERNELS_PFX})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(mainExe ${CMAKE_CURRENT_SOURCE_DIR}/Tucker_sthosvd.cpp)
target_link_libraries(mainExe kokkos_tucker Kokkos::kokkos Kokkos::kokkoskernels)
