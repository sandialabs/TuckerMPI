
set(testname test_sthosvd_2)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../compare_eigenvalues.py compare_eigenvalues.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../compare_core_tensor.py compare_core_tensor.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../ascii_to_binary.py ascii_to_binary.py COPYONLY)

configure_file(gold0.txt gold0.txt COPYONLY)
configure_file(gold1.txt gold1.txt COPYONLY)
configure_file(gold2.txt gold2.txt COPYONLY)
configure_file(goldCoreTensor.txt goldCoreTensor.txt COPYONLY)
configure_file(raw.txt raw.txt COPYONLY)
configure_file(2x6x3.txt 2x6x3.txt COPYONLY)
configure_file(paramfile.txt paramfile.txt COPYONLY)

add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=2x6x3.txt
  -DTENSORBINDATAFILE=2x6x3.bin
  -DSKIPROWS=3
  -DEXENAME=${exename}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DEIGVAL_COMPARISON_RELTOL=1e-9
  -DEIGVAL_COMPARISON_ABSTOL=1e-11
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)
