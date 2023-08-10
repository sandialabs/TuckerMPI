include(FindUnixCommands)

# create directory needed by drivers to output things
#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/compressed)

#
# 1. convert tensor data from txt file to binary
#
set(CMD "python3 ${PYTHON_COMPARE_SCRIPTS_DIR}/ascii_to_binary.py -i ${TENSORASCIIDATAFILE} -o ${TENSORBINDATAFILE} --skip ${SKIPROWS}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "binary convertion failed")
else()
  message("ascii-to-bin succeeded!")
endif()

#
# 2. run driver
#
if(DEFINED NUMPROC)
  set(CMD "${MPIEXEC} -n ${NUMPROC} ${EXENAME}")
  message(STATUS ${CMD})
  execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
  if(RES)
    message(FATAL_ERROR "${EXENAME} driver exe failed")
  else()
    message("${EXENAME} driver exe succeeded!")
  endif()

else()

  set(CMD "${EXENAME}")
  message(STATUS ${CMD})
  execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
  if(RES)
    message(FATAL_ERROR "${EXENAME} driver exe failed")
  else()
    message("${EXENAME} driver exe succeeded!")
  endif()

endif()


#
# 3. run comparison against gold
#
set(CMD "python3 ${PYTHON_COMPARE_SCRIPTS_DIR}/comparator.py")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "comparison failed")
else()
  message("comparison succeeded!")
endif()
