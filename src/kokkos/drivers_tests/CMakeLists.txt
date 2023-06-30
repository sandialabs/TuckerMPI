
add_subdirectory(onnode)

if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  add_subdirectory(mpi)
endif()
