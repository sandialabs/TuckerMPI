set(testname test_paramfile_1)
set(exename test_paramfile_1_exe)

configure_file(paramfile.txt paramfile.txt COPYONLY)

set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers/CmdLineParse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers/ParameterFileParserUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_executable(${exename} ${SOURCES})
target_include_directories(${exename} BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../drivers/onnode)
add_test(NAME ${testname} COMMAND ${exename})
set_tests_properties(${testname}
  PROPERTIES PASS_REGULAR_EXPRESSION "PASS"
  FAIL_REGULAR_EXPRESSION "FAILED"
  )
