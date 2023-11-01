include(FindUnixCommands)

#
# run streaming driver
#
set(CMD "${MPIEXEC} -n 4 ${TEST_EXE_DIR}/MpiKokkosTucker_streaming_sthosvd --parameter-file compress.txt")
message(STATUS ${CMD})
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Streaming driver failed")
endif()

#
# compute reconstruction
#
set(CMD "${MPIEXEC} -n 4 ${TEST_EXE_DIR}/MpiKokkosTucker_reconstruct --parameter-file reconstruct.txt")
message(STATUS ${CMD})
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Reconstruction driver failed")
endif()

#
# 5. run comparison of reconstruction against original
#
set(CMD "python3 ${TEST_SRC_DIR}/compute_errors.py \
                --config_dir ${TEST_BIN_DIR} \
                --rel_tol=1e-10 \
                --abs_tol=1e-10")
message(STATUS ${CMD})
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "comparison failed")
endif()
