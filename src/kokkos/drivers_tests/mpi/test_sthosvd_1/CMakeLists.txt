
set(testname test_sthosvd_1)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

configure_file(compare.py compare.py COPYONLY)
configure_file(gold.txt gold.txt COPYONLY)
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
