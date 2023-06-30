include(FindUnixCommands)

# run script taking txt file and dumping binary file
set(CMD "python3 ascii_to_binary_tensor_data.py -i ${TENSORASCIIDATAFILE} -o ${TENSORBINDATAFILE} --skip ${SKIPROWS}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "binary convertion failed")
else()
  message("comparison succeeded!")
endif()
