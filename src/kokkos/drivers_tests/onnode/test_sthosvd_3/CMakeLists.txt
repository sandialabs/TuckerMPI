
set(testname test_sthosvd_3)
set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../../${STHOSVD_DRIVER_NAME})

set(ScriptsPython compare_utils.py compare_eigenvalues.py ascii_to_binary.py)
foreach(fi IN LISTS ScriptsPython)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../scripts_python/${fi} ${fi} COPYONLY)
endforeach()

configure_file(gold0.txt gold0.txt COPYONLY)
configure_file(gold1.txt gold1.txt COPYONLY)
configure_file(gold2.txt gold2.txt COPYONLY)
configure_file(raw.txt raw.txt COPYONLY)
configure_file(4x4x4.txt 4x4x4.txt COPYONLY)
configure_file(paramfile.txt paramfile.txt COPYONLY)

add_test(NAME ${testname}
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=4x4x4.txt
  -DTENSORBINDATAFILE=4x4x4.bin
  -DSKIPROWS=3
  -DEXENAME=${exename}
  -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
  -DEIGVAL_COMPARISON_RELTOL=1e-9
  -DEIGVAL_COMPARISON_ABSTOL=1e-11
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)
