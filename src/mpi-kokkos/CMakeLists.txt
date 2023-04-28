
#INCLUDE_DIRECTORIES(../serial ${MPI_CXX_INCLUDE_PATH})
#FILE(GLOB SOURCES "*.cpp")
#ADD_LIBRARY(mpi_tucker SHARED ${SOURCES})
#TARGET_LINK_LIBRARIES(mpi_tucker serial_tucker ${MPI_CXX_LIBRARIES})
# ADD_SUBDIRECTORY(drivers)
# ADD_SUBDIRECTORY(tests)
# ADD_SUBDIRECTORY(converter)

set(SOURCES)
LIST(APPEND SOURCES 
  Tucker_IO_Util.cpp
  Tucker_MemManager.cpp
  Tucker_SizeArray.cpp
  TuckerMPI_ProcessorGrid.cpp
  TuckerMPI_Map.cpp
  TuckerMPI_Distribution.cpp
  )
ADD_LIBRARY(mpi_kokkos_tucker SHARED ${SOURCES})

find_package(KokkosKernels REQUIRED HINTS ${KOKKOSKERNELS_PFX})

include_directories(${MPI_CXX_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(mainExe ${CMAKE_CURRENT_SOURCE_DIR}/drivers/TuckerMPI_sthosvd.cpp)
target_link_libraries(mainExe mpi_kokkos_tucker Kokkos::kokkos Kokkos::kokkoskernels ${MPI_CXX_LIBRARIES})

