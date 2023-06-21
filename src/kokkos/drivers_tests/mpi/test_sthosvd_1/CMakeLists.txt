
set(testname test_sthosvd_1)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../compare_eigenvalues.py compare_eigenvalues.py COPYONLY)
configure_file(gold0.txt gold0.txt COPYONLY)
configure_file(gold1.txt gold1.txt COPYONLY)
configure_file(gold2.txt gold2.txt COPYONLY)
configure_file(gold3.txt gold3.txt COPYONLY)
configure_file(raw.txt raw.txt COPYONLY)
configure_file(3x5x7x11.mpi 3x5x7x11.mpi COPYONLY)
configure_file(paramfile.txt paramfile.txt COPYONLY)

add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DMPIEXEC=${MPIEXEC_EXECUTABLE}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DEXENAME=${exename}
  -DNUMPROC=8
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)
