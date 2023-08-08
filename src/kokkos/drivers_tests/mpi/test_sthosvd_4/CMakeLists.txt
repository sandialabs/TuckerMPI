
set(testname test_sthosvd_4)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

configure_file(gold0.txt gold0.txt COPYONLY)
configure_file(gold1.txt gold1.txt COPYONLY)
configure_file(gold2.txt gold2.txt COPYONLY)
configure_file(gold3.txt gold3.txt COPYONLY)
configure_file(gold_stats.txt gold_stats.txt COPYONLY)
configure_file(raw.txt raw.txt COPYONLY)
configure_file(gold_normalized_tensor.txt gold_normalized_tensor.txt COPYONLY)
configure_file(pre.txt pre.txt COPYONLY)
configure_file(3x5x7x11.txt 3x5x7x11.txt COPYONLY)
configure_file(paramfile.txt paramfile.txt COPYONLY)

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
  -DEIGVAL=0
  -DSTATS=1
  -DNORMALIZEDTENSOR=1
  -DSTATS_COMPARISON_RELTOL="1e-4"
  -DSTATS_COMPARISON_ABSTOL="1e-6"
  -DGOLDNORMALIZEDTENSORASCIIDATAFILE=gold_normalized_tensor.txt
  -DGOLDNORMALIZEDTENSORBINDATAFILE=gold_normalized_tensor.bin
  -DPREPROC_COMPARISON_RELTOL="1e-7"
  -DPREPROC_COMPARISON_ABSTOL="1e-8"
  -P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
)
