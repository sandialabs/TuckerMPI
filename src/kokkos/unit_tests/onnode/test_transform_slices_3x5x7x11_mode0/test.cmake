include(FindUnixCommands)

# run script taking txt file and dumping binary file
set(CMD "python3 ascii_to_binary.py -i ${TENSORASCIIDATAFILE} -o ${TENSORBINDATAFILE} --skip ${SKIPROWS}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "binary convertion failed")
else()
  message("ascii-to-bin succeeded!")
endif()