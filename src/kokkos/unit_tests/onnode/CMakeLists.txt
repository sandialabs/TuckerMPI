
set(GTEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/gtest_main.cc)

macro(add_utest_kokkos TESTNAME TESTSRCS)
  set(testNameFinal ${TESTNAME})
  add_executable(${testNameFinal} ${TESTSRCS} ${GTEST_MAIN})
  target_link_libraries(
    ${testNameFinal}
    ${KokkosOnlyTuckerLib} ${TUCKER_ALL_TPLS_NEEDED}
    GTest::gtest_main)

  add_test(NAME ${testNameFinal} COMMAND ${testNameFinal})
endmacro()


set(TEST1 test_tensor_traits)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)

set(TEST2 test_tensor)
add_utest_kokkos(${TEST2} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST2}.cc)

set(TEST3 test_tuckertensor_traits)
add_utest_kokkos(${TEST3} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST3}.cc)

set(TEST4 test_tuckertensor)
add_utest_kokkos(${TEST4} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST4}.cc)
target_compile_definitions(${TEST4} PUBLIC TUCKER_ALLOW_PRIVATE_CONSTRUCTORS_TO_BE_PUBLIC_FOR_TESTING)

set(TEST5 test_tensorgrameigenvalues_storage)
add_utest_kokkos(${TEST5} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST5}.cc)
target_compile_definitions(${TEST5} PUBLIC TUCKER_ALLOW_PRIVATE_CONSTRUCTORS_TO_BE_PUBLIC_FOR_TESTING)

set(TEST6 test_syrk_kokkos)
add_utest_kokkos(${TEST6} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST6}.cc)

set(TEST7 test_compute_slice_metrics)
add_utest_kokkos(${TEST7} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST7}.cc)

add_subdirectory(test_transform_slices_3x5x7x11)