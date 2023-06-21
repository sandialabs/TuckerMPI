include(FindUnixCommands)

# put here command to run script taking txt file and dumping binary file

# REMOVE OVERSUBSCRIBE HERE

set(CMD "${MPIEXEC} --oversubscribe -n ${NUMPROC} ${EXENAME}")
message(STATUS ${CMD})
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "run failed")
else()
  message("run succeeded!")
endif()

set(CMD "python3 compare.py")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "comparison failed")
else()
  message("comparison succeeded!")
endif()
