include(FindUnixCommands)

# convert .txt into .bin files by running test.cmake
set(SCRIPT_PY ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers_tests/python_scripts/ascii_to_binary.py)
list(APPEND TXT_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_mm0.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_mm1.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_mm2.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_mm3.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_sc0.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_sc1.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_sc2.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_sc3.txt
)
list(APPEND BIN_OUTPUT_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_mm0.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_mm1.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_mm2.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_mm3.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_sc0.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_sc1.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_sc2.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_sc3.bin
)

foreach(fi bi IN ZIP_LISTS TXT_SOURCE_FILES BIN_OUTPUT_FILES)
    set(CMD "python3 ${SCRIPT_PY} -i ${fi} -o ${bi} --skip 3")
    execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
endforeach()

# run test
foreach(nprocs RANGE 1 8)
  set(T1 test_normalize)
  add_utest_mpikokkos(${T1} ${CMAKE_CURRENT_SOURCE_DIR}/${T1}.cc ${nprocs})
endforeach()