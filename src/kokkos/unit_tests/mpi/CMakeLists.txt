include(FindUnixCommands)

set(SCRIPT_PY ${CMAKE_CURRENT_SOURCE_DIR}/../../drivers_tests/python_scripts/ascii_to_binary.py)

set(GTEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/gtest_main.cc)

macro(add_utest_mpikokkos TESTNAME TESTSRCS nRANKS)
  set(testNameFinal ${TESTNAME}_np${nRANKS})
  add_executable(${testNameFinal} ${TESTSRCS} ${GTEST_MAIN})

  target_link_libraries(
    ${testNameFinal}
    ${MpiKokkosTuckerLib}
    ${TUCKER_ALL_TPLS_NEEDED} # this contains already kokkos and mpi libs
    GTest::gtest_main)

  add_test(
    NAME ${testNameFinal}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nRANKS}
    ${MPIEXEC_PREFLAGS} ${testNameFinal} ${MPIEXEC_POSTFLAGS}
    )
endmacro()

# convert all txt files containing tensor data into binary
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tensor_data_files)
list(APPEND TXT_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_data_files/3x5x7x11.txt
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_data_files/4x4x4.txt
)
list(APPEND BIN_OUTPUT_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/tensor_data_files/3x5x7x11.bin
  ${CMAKE_CURRENT_BINARY_DIR}/tensor_data_files/4x4x4.bin
)
foreach(fi bi IN ZIP_LISTS TXT_SOURCE_FILES BIN_OUTPUT_FILES)
  set(CMD "python3 ${SCRIPT_PY} -i ${fi} -o ${bi} --skip 3")
  message(STATUS ${CMD})
  execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
  if(RES)
    message(FATAL_ERROR "binary convertion of tensor_data_files failed")
  else()
    message("ascii-to-bin for tensor_data_files succeeded!")
  endif()
endforeach()


set(T1 test_map)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 3)

set(T1 test_comms_equality)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 4)

set(T1 test_ttm_pack)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 2)

set(T1 test_compute_slice_metrics)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 3)

set(T1 test_local_rank_k_for_gram)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 4)

set(T1 test_new_gram)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 2)

set(T1 test_new_gram)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 6)

set(T1 test_big_gram_0)
add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc 8)

foreach(nprocs RANGE 2 8)
  set(T1 test_norm)
  add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc ${nprocs})
endforeach()

add_subdirectory(test_normalize_3x5x7x11)