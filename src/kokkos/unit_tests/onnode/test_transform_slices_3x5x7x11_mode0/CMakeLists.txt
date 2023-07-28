include(FindUnixCommands)

# convert .txt files by running test.cmake
set(SCRIPT_PY   ${CMAKE_CURRENT_SOURCE_DIR}/../../ascii_to_binary.py)
list(APPEND FILES_TXT
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_ss0.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_ss1.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_ss2.txt
    ${CMAKE_CURRENT_SOURCE_DIR}/gold_3x5x7x11_ss3.txt
)
list(APPEND BIN_TXT
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_ss0.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_ss1.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_ss2.bin
    ${CMAKE_CURRENT_BINARY_DIR}/gold_3x5x7x11_ss3.bin
)

foreach(fi bi IN ZIP_LISTS FILES_TXT BIN_TXT)
    set(CMD "python3 ${SCRIPT_PY} -i ${fi} -o ${bi} --skip 3")
    execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
endforeach()

# run test
set(TEST1 test_transform_slices)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc ../gtest_main.cc)