INCLUDE_DIRECTORIES(.. ../../serial)

ADD_EXECUTABLE(reconstruct TuckerMPI_reconstruct.cpp)
TARGET_LINK_LIBRARIES(reconstruct mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(sthosvd TuckerMPI_sthosvd.cpp)
TARGET_LINK_LIBRARIES(sthosvd mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})