include(FindUnixCommands)

if(MPIEXEC)
  set(MPICMD "${MPIEXEC} -n ${NUMPROCS} ${MPIEXEC_PREFLAGS}")
  set(IMPL "MpiKokkosTucker")
else()
  set(MPICMD "")
  set(IMPL "KokkosTucker")
endif()

#
# run streaming driver
#
set(CMD "${MPICMD} ${TEST_EXE_DIR}/${IMPL}_streaming_sthosvd --parameter-file compress.txt")
message(STATUS ${CMD})
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "Streaming driver failed")
endif()

#
# compute reconstruction
#
set(CMD "${MPICMD} ${TEST_EXE_DIR}/${IMPL}_reconstruct --parameter-file reconstruct.txt")
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
