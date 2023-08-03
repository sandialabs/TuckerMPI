configure_file(paramfile_1.txt paramfile_1.txt COPYONLY)
configure_file(paramfile_2.txt paramfile_2.txt COPYONLY)

set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers/CmdLineParse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers/ParameterFileParserUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

set(testname test_paramfile_1)
set(exename test_paramfile_1_exe)
add_executable(${exename} ${SOURCES})
target_compile_definitions(${exename} PUBLIC -DPARAM1)
target_include_directories(${exename} BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers/onnode)
add_test(NAME ${testname} COMMAND ${exename})
set_tests_properties(${testname}
  PROPERTIES PASS_REGULAR_EXPRESSION "PASS"
  FAIL_REGULAR_EXPRESSION "FAILED"
  )

set(testname test_paramfile_2)
set(exename test_paramfile_2_exe)
add_executable(${exename} ${SOURCES})
target_compile_definitions(${exename} PUBLIC -DPARAM2)
target_include_directories(${exename} BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers/onnode)
add_test(NAME ${testname} COMMAND ${exename})
set_tests_properties(${testname}
  PROPERTIES PASS_REGULAR_EXPRESSION "PASS"
  FAIL_REGULAR_EXPRESSION "FAILED"
  )
