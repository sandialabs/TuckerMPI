INCLUDE_DIRECTORIES(..)

ADD_EXECUTABLE(Tucker_reconstruct Tucker_reconstruct.cpp)
TARGET_LINK_LIBRARIES(Tucker_reconstruct serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(Tucker_sthosvd Tucker_sthosvd.cpp)
TARGET_LINK_LIBRARIES(Tucker_sthosvd serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})