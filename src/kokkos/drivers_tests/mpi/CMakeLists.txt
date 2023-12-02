
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


set(exename ${CMAKE_CURRENT_BINARY_DIR}/../../${DISTRIBUTED_STHOSVD_DRIVER_NAME})

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
FOREACH(subdir ${SUBDIRS})
  message(STATUS "Preparing test for = ${subdir}")
  set(TEST_RUN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${subdir})
  set(TEST_NAME ${subdir})

  file(MAKE_DIRECTORY ${TEST_RUN_DIR})

  file(GLOB SRCS ${subdir}/*.txt)
  FOREACH(ftxt ${SRCS})
    configure_file(${ftxt} ${TEST_RUN_DIR}/${fi} COPYONLY)
  ENDFOREACH()

  file(STRINGS "${subdir}/num_procs.txt" NUMPROCS)

  add_test(NAME ${TEST_NAME}
    WORKING_DIRECTORY ${TEST_RUN_DIR}
    COMMAND ${CMAKE_COMMAND}
    -DTENSORASCIIDATAFILE=${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/tensor_data.txt
    -DTENSORBINDATAFILE=${TEST_RUN_DIR}/tensor_data.bin
    -DSKIPROWS=3
    -DEXENAME=${exename}
    -DMPIEXEC=${MPIEXEC_EXECUTABLE}
    -DOUTDIR=${TEST_RUN_DIR}
    -DPYTHON_COMPARE_SCRIPTS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../python_scripts
    -DNUMPROC=${NUMPROCS}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
  )
ENDFOREACH()
