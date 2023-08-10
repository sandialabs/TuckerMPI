
set(testname test_sthosvd_3)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

set(TXT_FILE_TO_COPY
  gold0.txt gold1.txt gold2.txt goldCoreTensor.txt
  goldFactorMatrix0.txt goldFactorMatrix1.txt goldFactorMatrix2.txt
  raw.txt 4x4x4.txt paramfile.txt
)
foreach(fi IN LISTS TXT_FILE_TO_COPY)
  configure_file(${fi} ${fi} COPYONLY)
endforeach()

add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=4x4x4.txt
  -DTENSORBINDATAFILE=4x4x4.bin
  -DSKIPROWS=3
  -DEXENAME=${exename}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DPYTHON_COMPARE_SCRIPTS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../python_scripts
  -DEIGVAL_COMPARISON_RELTOL=1e-8
  -DEIGVAL_COMPARISON_ABSTOL=1e-8
  -DCORETENSOR_COMPARISON_RELTOL=1e-8
  -DCORETENSOR_COMPARISON_ABSTOL=1e-8
  -DFACTORMATRIX_COMPARISON_RELTOL=1e-8
  -DFACTORMATRIX_COMPARISON_ABSTOL=1e-8
  -P ${CMAKE_CURRENT_SOURCE_DIR}/../../test.cmake
)
