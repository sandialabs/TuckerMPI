set(testname test_sthosvd_2)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

configure_file(gold0.txt gold0.txt COPYONLY)
configure_file(gold1.txt gold1.txt COPYONLY)
configure_file(gold2.txt gold2.txt COPYONLY)
configure_file(gold_stats.txt gold_stats.txt COPYONLY)
configure_file(raw.txt raw.txt COPYONLY)
configure_file(2x6x3.txt 2x6x3.txt COPYONLY)
configure_file(paramfile.txt paramfile.txt COPYONLY)

add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=2x6x3.txt
  -DTENSORBINDATAFILE=2x6x3.bin
  -DSKIPROWS=3
  -DEXENAME=${exename}
  -DMPIEXEC=${MPIEXEC_EXECUTABLE}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DPYTHON_COMPARE_SCRIPTS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../python_scripts
  -DNUMPROC=8
  -DEIGVAL_COMPARISON_RELTOL="1e-9"
  -DEIGVAL_COMPARISON_ABSTOL="1e-11"
  -DSTATS_COMPARISON_RELTOL="1e-4"
  -DSTATS_COMPARISON_ABSTOL="1e-6"
  -P ${CMAKE_CURRENT_SOURCE_DIR}/../test_compare_eigval_and_stats.cmake
)
