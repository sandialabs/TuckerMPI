
if(TUCKER_IMPL_ENABLE_KOKKOSONLY)
  add_subdirectory(onnode)
endif()

if(TUCKER_IMPL_ENABLE_MPIKOKKOS)
  add_subdirectory(mpi)
endif()
