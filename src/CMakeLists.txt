CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(Tucker C CXX)

#=====================================================================
# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed.
    Please make a new directory (called a build directory) and run CMake from there.
    You may need to remove CMakeCache.txt.")
endif()

#=====================================================================
# check build type
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# convert cmake build type to lower string
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if( NOT cmake_build_type_tolower STREQUAL "debug"
    AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\".
    Allowed values are Debug, Release (case-insensitive).")
endif()

#=====================================================================

# public keywords
option(TUCKER_ENABLE_KOKKOS OFF)
option(TUCKER_ENABLE_MPI OFF)

# internal keywords
option(TUCKER_IMPL_ENABLE_KOKKOSONLY OFF)
option(TUCKER_IMPL_ENABLE_MPIKOKKOS OFF)

if(TUCKER_ENABLE_KOKKOS AND NOT TUCKER_ENABLE_MPI)
  SET(TUCKER_IMPL_ENABLE_KOKKOSONLY ON)
elseif(TUCKER_ENABLE_KOKKOS AND TUCKER_ENABLE_MPI)
  SET(TUCKER_IMPL_ENABLE_MPIKOKKOS ON)
endif()

IF(TUCKER_IMPL_ENABLE_KOKKOSONLY OR TUCKER_IMPL_ENABLE_MPIKOKKOS)
  SET(CMAKE_CXX_STANDARD 17)
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORT_TARGET_CPP17)
  if(NOT COMPILER_SUPPORT_TARGET_CPP17)
    message(FATAL_ERROR
      "C++ compiler does not support -std=c++17. This is required for building kokkos-only.")
  endif()
  MESSAGE(STATUS "Setting CMAKE_CXX_STANDARD=17")

  enable_testing()
  ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/kokkos)

ELSE()
  SET(CMAKE_CXX_STANDARD 11)
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x")

  # SET(BLA_VENDOR Intel10_64lp_seq)
  FIND_PACKAGE(BLAS REQUIRED)
  FIND_PACKAGE(LAPACK REQUIRED)
  FIND_PACKAGE(MPI)

  SET(EXECUTABLE_OUTPUT_PATH bin)

  INCLUDE(CTest)
  ENABLE_TESTING()

  ADD_SUBDIRECTORY(serial)
  IF(MPI_CXX_FOUND)
    ADD_SUBDIRECTORY(mpi)
  ENDIF()

ENDIF()
