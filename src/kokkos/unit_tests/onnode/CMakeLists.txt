
macro(add_utest_kokkos TESTNAME TESTSRCS GTEST_PATH)
  set(testNameFinal ${TESTNAME})
  add_executable(${testNameFinal} ${TESTSRCS} ${GTEST_PATH})
  target_link_libraries(
    ${testNameFinal}
    ${KokkosOnlyTuckerLib} ${TUCKER_ALL_TPLS_NEEDED}
    GTest::gtest_main)

  add_test(NAME ${testNameFinal} COMMAND ${testNameFinal})
endmacro()

set(GTEST_MAIN gtest_main.cc)

set(TEST1 test_tensor_traits)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ${GTEST_MAIN})
set(TEST1 test_tensor)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ${GTEST_MAIN})

set(TEST1 test_tuckertensor_traits)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ${GTEST_MAIN})
set(TEST1 test_tuckertensor)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ${GTEST_MAIN})

set(TEST1 test_tensorgrameigenvalues_storage)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ${GTEST_MAIN})

set(TEST1 test_syrk_kokkos)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ${GTEST_MAIN})

set(TEST1 test_compute_slice_metrics)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ${GTEST_MAIN})

add_subdirectory(test_transform_slices_3x5x7x11)