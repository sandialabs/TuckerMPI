INCLUDE_DIRECTORIES(..)

ADD_EXECUTABLE(compare Tucker_compare.cpp)
TARGET_LINK_LIBRARIES(compare serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(qr_routine_comp LAPACK_qr_routine_comp.cpp)
TARGET_LINK_LIBRARIES(qr_routine_comp serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(LAPACK_BLAS_routine_comp LAPACK_BLAS_routine_comp.cpp)
TARGET_LINK_LIBRARIES(LAPACK_BLAS_routine_comp serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

ADD_EXECUTABLE(qr_routine_comp_single LAPACK_qr_routine_comp.cpp)
TARGET_LINK_LIBRARIES(qr_routine_comp_single serial_tucker ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

SET_TARGET_PROPERTIES(
  qr_routine_comp_single
  PROPERTIES 
        COMPILE_DEFINITIONS DRIVER_SINGLE)