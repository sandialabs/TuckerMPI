include(FindUnixCommands)

if(TUCKER_IMPL_ENABLE_KOKKOSONLY)
  add_subdirectory(onnode)
endif()

if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  add_subdirectory(mpi)
endif()

#
# streaming sthosvd tests
#

if(TUCKER_IMPL_ENABLE_KOKKOSONLY OR TUCKER_IMPL_ENABLE_MPIKOKKOS)
  set(NUMPROCS_LIST "1")
  if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
    list(APPEND NUMPROCS_LIST "4")
  endif()
  if(TUCKER_IMPL_ENABLE_KOKKOSONLY)
    set(IMPL "onnode")
  else()
    set(IMPL "mpi")
  endif()
  foreach(NUMPROCS ${NUMPROCS_LIST})
    set(SUBDIR "test_streaming_sthosvd_n${NUMPROCS}")
    message(STATUS "Preparing test for ${SUBDIR}")
    set(TEST_RUN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${IMPL}/${SUBDIR})
    set(TEST_NAME ${SUBDIR})

    file(MAKE_DIRECTORY ${TEST_RUN_DIR})

    # create data
    set(CMD "python3 ${CMAKE_CURRENT_SOURCE_DIR}/sine_wave/generate_data.py \
                --mode_sizes 10 10 50 \
                --maximum_frequencies 1 1 1 \
                --frequency_bumps 1 1 1 \
                --noise_to_signal_ratio 0.0 \
                --data_dir ${TEST_RUN_DIR}/data")
    execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
    if(RES)
      message(FATAL_ERROR "generating data failed")
    endif()

    # create configuration files
    set(CMD "python3 ${CMAKE_CURRENT_SOURCE_DIR}/sine_wave/generate_conf.py \
                --mode_sizes 10 10 50 \
                --num_initial 10 \
                --tolerance 1e-3 \
                --num_proc ${NUMPROCS} \
                --data_dir ${TEST_RUN_DIR}/data \
                --config_dir ${TEST_RUN_DIR}")
    execute_process(COMMAND ${BASH} -c ${CMD} RESULT_VARIABLE RES)
    if(RES)
      message(FATAL_ERROR "generating conf files failed")
    endif()

    add_test(NAME ${TEST_NAME}
      WORKING_DIRECTORY ${TEST_RUN_DIR}
      COMMAND ${CMAKE_COMMAND}
      -D TEST_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/sine_wave
      -D TEST_BIN_DIR=${TEST_RUN_DIR}
      -D TEST_EXE_DIR=${CMAKE_CURRENT_BINARY_DIR}/..
      -D MPIEXEC=${MPIEXEC_EXECUTABLE}
      -D MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
      -D NUMPROCS=${NUMPROCS}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/sine_wave/test.cmake
      )
  endforeach()
endif()
