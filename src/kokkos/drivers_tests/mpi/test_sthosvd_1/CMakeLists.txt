
set(testname test_sthosvd_1)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

set(TXT_FILE_TO_COPY
  raw.txt 3x5x7x11.txt paramfile.txt
  gold0.txt gold1.txt gold2.txt gold3.txt goldCoreTensor.txt
  goldFactorMatrix0.txt goldFactorMatrix1.txt goldFactorMatrix2.txt goldFactorMatrix3.txt
  gold_stats.txt
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
  -DMPIEXEC=${MPIEXEC_EXECUTABLE}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DPYTHON_COMPARE_SCRIPTS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../python_scripts
  -DNUMPROC=8
  -DEIGVAL=1
  -DSTATS=1
  -DCORETENSOR=1
  -DFACTORS=1
  -DEIGVAL_COMPARISON_RELTOL="1e-9"
  -DEIGVAL_COMPARISON_ABSTOL="1e-11"
  -DSTATS_COMPARISON_RELTOL="1e-4"
  -DSTATS_COMPARISON_ABSTOL="1e-6"
  -DCORETENSOR_COMPARISON_RELTOL=1e-8
  -DCORETENSOR_COMPARISON_ABSTOL=1e-8
  -DFACTORMATRIX_COMPARISON_RELTOL=1e-8
  -DFACTORMATRIX_COMPARISON_ABSTOL=1e-8
  -P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
)
