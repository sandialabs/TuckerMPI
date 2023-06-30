
set(testname test_sthosvd_1)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../compare_eigenvalues.py compare_eigenvalues.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../ascii_to_binary_tensor_data.py ascii_to_binary_tensor_data.py COPYONLY)

configure_file(gold0.txt gold0.txt COPYONLY)
configure_file(gold1.txt gold1.txt COPYONLY)
configure_file(gold2.txt gold2.txt COPYONLY)
configure_file(gold3.txt gold3.txt COPYONLY)
configure_file(raw.txt raw.txt COPYONLY)
configure_file(3x5x7x11.txt 3x5x7x11.txt COPYONLY)
configure_file(paramfile.txt paramfile.txt COPYONLY)

add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=3x5x7x11.txt
  -DTENSORBINDATAFILE=3x5x7x11.mpi
  -DSKIPROWS=3
  -DEXENAME=${exename}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)
