
macro(add_utest_kokkos TESTNAME TESTSRCS)
  set(testNameFinal ${TESTNAME})
  add_executable(${testNameFinal} ${TESTSRCS} gtest_main.cc)
  target_link_libraries(
    ${testNameFinal}
    ${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED}
    GTest::gtest_main)

  add_test(NAME ${testNameFinal} COMMAND ${testNameFinal})
endmacro()

set(TEST1 test_tensor_traits)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)
set(TEST1 test_tensor)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)

set(TEST1 test_tuckertensor_traits)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)
set(TEST1 test_tuckertensor)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)

set(TEST1 test_tensorgrameigenvalues_storage)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)

set(TEST1 test_syrk_kokkos)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)

set(TEST1 test_compute_slice_metrics)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)

# set(TEST1 tensor)
# add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)
# set(TEST2 tucker_kokkosonly_ttm_using_kernels_gemm)
# add_utest_kokkos(${TEST2} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST2}.cc)
