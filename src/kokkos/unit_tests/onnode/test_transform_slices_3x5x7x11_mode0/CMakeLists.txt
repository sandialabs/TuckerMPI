include(FindUnixCommands)

# convert .txt files by running test.cmake
set(CMD "python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../ascii_to_binary.py -i ${CMAKE_CURRENT_SOURCE_DIR}/3x5x7x11.txt -o ${CMAKE_CURRENT_BINARY_DIR}/3x5x7x11.bin --skip 3")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)

set(CMD "python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../ascii_to_binary.py -i ${CMAKE_CURRENT_SOURCE_DIR}/3x5x7x11_ss0.txt -o ${CMAKE_CURRENT_BINARY_DIR}/3x5x7x11_ss0.bin --skip 3")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
# for loop: => 2 files

# run test
set(TEST1 test_transform_slices)
add_utest_kokkos(${TEST1} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST1}.cc)
