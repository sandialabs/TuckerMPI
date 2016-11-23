INCLUDE_DIRECTORIES(.. ../../serial)
ADD_SUBDIRECTORY(SC2016)

ADD_EXECUTABLE(partial_reconstruct partial_reconstruct.cpp)
TARGET_LINK_LIBRARIES(partial_reconstruct mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(mpi_driver driver.cpp)
TARGET_LINK_LIBRARIES(mpi_driver mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(reconstruct reconstruct.cpp)
TARGET_LINK_LIBRARIES(reconstruct mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(parallel_tensor_test parallel_tensor_test.cpp)
TARGET_LINK_LIBRARIES(parallel_tensor_test mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(text_to_binary text_to_binary.cpp)
TARGET_LINK_LIBRARIES(text_to_binary mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(mpi_test mpi_test.cpp)
TARGET_LINK_LIBRARIES(mpi_test mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(parallel_file_driver file_driver.cpp)
TARGET_LINK_LIBRARIES(parallel_file_driver mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(mpi_file_driver mpi_file_driver.cpp)
TARGET_LINK_LIBRARIES(mpi_file_driver mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(parallel_mpi_test parallel_mpi_test.cpp)
TARGET_LINK_LIBRARIES(parallel_mpi_test mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})