INCLUDE_DIRECTORIES(.. ../../serial)

ADD_EXECUTABLE(test_input test.cpp)
TARGET_LINK_LIBRARIES(test_input mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(test_input_single test.cpp)
TARGET_LINK_LIBRARIES(test_input_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(reconstruct TuckerMPI_reconstruct.cpp)
TARGET_LINK_LIBRARIES(reconstruct mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(general_reconstruct TuckerMPI_general_reconstruct.cpp)
TARGET_LINK_LIBRARIES(general_reconstruct mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(sthosvd TuckerMPI_sthosvd.cpp)
TARGET_LINK_LIBRARIES(sthosvd mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(sthosvd_test TuckerMPI_sthosvd_test.cpp)
TARGET_LINK_LIBRARIES(sthosvd_test mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(scaling TuckerMPI_scaling.cpp)
TARGET_LINK_LIBRARIES(scaling mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(generate TuckerMPI_generate.cpp)
TARGET_LINK_LIBRARIES(generate mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(gram TuckerMPI_gram.cpp)
TARGET_LINK_LIBRARIES(gram mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(LQ TuckerMPI_LQ.cpp)
TARGET_LINK_LIBRARIES(LQ mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

#single precision version
ADD_EXECUTABLE(reconstruct_single TuckerMPI_reconstruct.cpp)
TARGET_LINK_LIBRARIES(reconstruct_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(general_reconstruct_single TuckerMPI_general_reconstruct.cpp)
TARGET_LINK_LIBRARIES(general_reconstruct_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(sthosvd_single TuckerMPI_sthosvd.cpp)
TARGET_LINK_LIBRARIES(sthosvd_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(sthosvd_test_single TuckerMPI_sthosvd_test.cpp)
TARGET_LINK_LIBRARIES(sthosvd_test_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(scaling_single TuckerMPI_scaling.cpp)
TARGET_LINK_LIBRARIES(scaling_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(generate_single TuckerMPI_generate.cpp)
TARGET_LINK_LIBRARIES(generate_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(gram_single TuckerMPI_gram.cpp)
TARGET_LINK_LIBRARIES(gram_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(LQ_single TuckerMPI_LQ.cpp)
TARGET_LINK_LIBRARIES(LQ_single mpi_tucker serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

SET_TARGET_PROPERTIES(
  test_input_single
  reconstruct_single
  general_reconstruct_single
  sthosvd_single
  sthosvd_test_single
  scaling_single
  generate_single
  gram_single
  LQ_single
  PROPERTIES 
        COMPILE_DEFINITIONS DRIVER_SINGLE)