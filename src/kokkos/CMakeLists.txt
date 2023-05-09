CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(kokkos)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
# GoogleTest

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

# GoogleTest
enable_testing()

add_executable(
  hello_test
  AM_GoogleTests.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hello_test)
# GoogleTest