INCLUDE_DIRECTORIES(../.. ../../../serial)

ADD_EXECUTABLE(gram_driver gram_driver.cpp)
TARGET_LINK_LIBRARIES(gram_driver mpi_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})