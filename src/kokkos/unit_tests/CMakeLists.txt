
# try to find Gtest
find_package(GTest QUIET)
if (GTEST_FOUND)
  message(STATUS "\nGTest was found.")
  message(STATUS "GTEST_INCLUDE_DIRS = ${GTEST_INCLUDE_DIRS}")
  message(STATUS "GTEST_LIBRARIES = ${GTEST_LIBRARIES}\n")
else()
  message(STATUS "GTest not found, fetching it.")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

if(TUCKER_IMPL_ENABLE_KOKKOSONLY)
  add_subdirectory(onnode)
endif()

if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  add_subdirectory(mpi)
endif()
