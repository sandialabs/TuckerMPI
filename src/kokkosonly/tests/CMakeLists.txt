
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

macro(add_utest_kokkos TESTNAME TESTSRCS)
  set(testNameFinal ${TESTNAME})
  add_executable(${testNameFinal} ${TESTSRCS} gtest_main.cc)
  target_link_libraries(
    ${testNameFinal}
    ${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED}
    GTest::gtest_main)

  add_test(NAME ${testNameFinal} COMMAND ${testNameFinal})
endmacro()

set(TEST1 tucker_kokkosonly_tensor)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)

set(TEST2 tucker_kokkosonly_ttm_using_kernels_gemm)
add_utest_kokkos(${TEST2} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST2}.cc)

set(TEST3 tucker_kokkosonly_ttm_1dtensor)
add_utest_kokkos(${TEST3} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST3}.cc)