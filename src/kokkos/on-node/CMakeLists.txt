set(SHARED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../shared)

set(KokkosOnlyTuckerLib kokkosonly_tucker)
set(SOURCES)
LIST(APPEND SOURCES
  ${SHARED_SRC}/Tucker_BlasWrapper.cpp
  ${SHARED_SRC}/Tucker_BoilerPlate_IO.cpp
  ${SHARED_SRC}/Tucker_CmdLineParse.cpp
  ${SHARED_SRC}/Tucker_ParameterFileParserUtils.cpp
)
ADD_LIBRARY(${KokkosOnlyTuckerLib} SHARED ${SOURCES})

find_package(KokkosKernels REQUIRED)
set(ALL_TPLS_NEEDED
  Kokkos::BLAS Kokkos::LAPACK Kokkos::kokkos Kokkos::kokkoskernels)
TARGET_LINK_LIBRARIES(${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED})

include_directories(${SHARED_SRC} ${CMAKE_CURRENT_SOURCE_DIR})

# add tests
add_subdirectory(tests)

#
# create executable for sthosvd
#
set(STHOSVD_SOURCES)
LIST(APPEND STHOSVD_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/drivers/TuckerOnNode_sthosvd.cpp
)
add_executable(Tucker_sthosvd ${STHOSVD_SOURCES})
target_link_libraries(Tucker_sthosvd ${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED})
