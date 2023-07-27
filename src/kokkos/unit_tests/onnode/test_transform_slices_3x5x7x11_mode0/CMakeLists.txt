set(testname test_transform_slices_3x5x7x11_mode0)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../ascii_to_binary.py ascii_to_binary.py COPYONLY)

configure_file(3x5x7x11.txt 3x5x7x11.txt COPYONLY)
configure_file(3x5x7x11_ss0.txt 3x5x7x11_ss0.txt COPYONLY)

# convert .txt files by running test.cmake
execute_process(
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=3x5x7x11.txt
  -DTENSORBINDATAFILE=3x5x7x11.bin
  -DSKIPROWS=3
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)
execute_process(
  COMMAND ${CMAKE_COMMAND}
  -DTENSORASCIIDATAFILE=3x5x7x11_ss0.txt
  -DTENSORBINDATAFILE=3x5x7x11_ss0.bin
  -DSKIPROWS=3
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)

# run test
set(testNameFinal ${testname})
add_executable(
    ${testNameFinal}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_transform_slices.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../gtest_main.cc)
target_link_libraries(
    ${testNameFinal}
    ${KokkosOnlyTuckerLib} ${ALL_TPLS_NEEDED}
    GTest::gtest_main)
add_test(NAME ${testNameFinal} COMMAND ${testNameFinal})
