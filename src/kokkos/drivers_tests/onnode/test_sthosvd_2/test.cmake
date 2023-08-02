include(FindUnixCommands)

# run script taking txt file and dumping binary file
set(CMD "python3 ascii_to_binary.py -i ${TENSORASCIIDATAFILE} -o ${TENSORBINDATAFILE} --skip ${SKIPROWS}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "binary convertion failed")
else()
  message("ascii-to-bin succeeded!")
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/compressed)

set(CMD "${EXENAME}")
message(STATUS ${CMD})
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "run failed")
else()
  message("run succeeded!")
endif()

set(CMD "python3 compare_eigenvalues.py --rtol ${EIGVAL_COMPARISON_RELTOL} --atol ${EIGVAL_COMPARISON_ABSTOL}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "eigenvalues comparison failed")
else()
  message("eigenvalues comparison succeeded!")
endif()

set(CMD "python3 compare_core_tensor.py --rtol ${EIGVAL_COMPARISON_RELTOL} --atol ${EIGVAL_COMPARISON_ABSTOL}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "core tensor comparison failed")
else()
  message("core tensor comparison succeeded!")
endif()

set(CMD "python3 compare_factor_matrices.py --rtol ${EIGVAL_COMPARISON_RELTOL} --atol ${EIGVAL_COMPARISON_ABSTOL}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "factor matrices comparison failed")
else()
  message("factor matrices comparison succeeded!")
endif()