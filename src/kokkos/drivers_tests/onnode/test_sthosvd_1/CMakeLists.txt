
set(testname test_sthosvd_1)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

set(ScriptsPython
  ascii_to_binary.py
  compare_utils.py compare_eigenvalues.py
  compare_core_tensor.py compare_factor_matrices.py)
foreach(fi IN LISTS ScriptsPython)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../python_scripts/${fi} ${fi} COPYONLY)
endforeach()

set(TXT_FILE_TO_COPY
  gold0.txt gold1.txt gold2.txt gold3.txt
  goldCoreTensor.txt
  goldFactorMatrice0.txt goldFactorMatrice1.txt goldFactorMatrice2.txt goldFactorMatrice3.txt
  raw.txt 3x5x7x11.txt paramfile.txt
)
foreach(fi IN LISTS TXT_FILE_TO_COPY)
  configure_file(${fi} ${fi} COPYONLY)
endforeach()

add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=3x5x7x11.txt
  -DTENSORBINDATAFILE=3x5x7x11.bin
  -DSKIPROWS=3
  -DEXENAME=${exename}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DEIGVAL_COMPARISON_RELTOL=1e-9
  -DEIGVAL_COMPARISON_ABSTOL=1e-11
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)
