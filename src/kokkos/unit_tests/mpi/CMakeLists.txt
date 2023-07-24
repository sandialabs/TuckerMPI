
macro(add_utest_mpikokkos TESTNAME TESTSRCS nRANKS)

  set(testNameFinal ${TESTNAME}_np${nRANKS})
  add_executable(${testNameFinal} ${TESTSRCS} gtest_main.cc)

  target_link_libraries(
    ${testNameFinal}
    ${MpiKokkosTuckerLib}
    ${ALL_TPLS_NEEDED} # this contains already kokkos and mpi libs
    GTest::gtest_main)

  add_test(
    NAME ${testNameFinal}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nRANKS}
    ${MPIEXEC_PREFLAGS} ${testNameFinal} ${MPIEXEC_POSTFLAGS}
    )
endmacro()

set(T1 test_map)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 3)

set(T1 test_comms_equality)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 4)

set(T1 test_pack_data)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 2)

set(T1 test_pack_data2)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 2)

set(T1 test_pack_data3)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 2)