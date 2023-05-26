set(KokkosOnlyTuckerLib kokkosonly_tucker)
set(SOURCES)
LIST(APPEND SOURCES
  Tucker_BlasWrapper.cpp
  Tucker_SizeArray.cpp
)
ADD_LIBRARY(${KokkosOnlyTuckerLib} SHARED ${SOURCES})

find_package(KokkosKernels REQUIRED)
set(ALL_TPLS_NEEDED
  Kokkos::BLAS Kokkos::LAPACK Kokkos::kokkos Kokkos::kokkoskernels)
TARGET_LINK_LIBRARIES(${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add tests
add_subdirectory(tests)

#
# create executable for sthosvd
#
set(STHOSVD_SOURCES)
LIST(APPEND STHOSVD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/Tucker_CmdLineParse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/Tucker_ParameterFileParserUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/Tucker_sthosvd.cpp
)
add_executable(Tucker_sthosvd ${STHOSVD_SOURCES})
target_link_libraries(Tucker_sthosvd ${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED})
