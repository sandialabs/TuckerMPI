include(FindUnixCommands)

message(STATUS "Preparing test for test_streaming_sthosvd_n4")

#
# create data
#
set(CMD "python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_data.py \
                --mode_sizes 10 10 50 \
                --maximum_frequencies 1 1 1 \
                --noise_to_signal_ratio 0.0 \
                --data_dir ${CMAKE_CURRENT_BINARY_DIR}/data")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "generating data failed")
endif()

#
# create configuration files
#
set(CMD "python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_conf.py \
                --mode_sizes 10 10 50 \
                --num_initial 10 \
                --tolerance 1e-3 \
                --num_proc 4 \
                --data_dir ${CMAKE_CURRENT_BINARY_DIR}/data \
                --config_dir ${CMAKE_CURRENT_BINARY_DIR}")
execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
if(RES)
  message(FATAL_ERROR "generating conf files failed")
endif()

add_test(NAME test_streaming_sthosvd_n4
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND}
    -D TEST_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -D TEST_BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -D TEST_EXE_DIR=${CMAKE_CURRENT_BINARY_DIR}/../../../
    -D MPIEXEC=${MPIEXEC_EXECUTABLE}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
  )
