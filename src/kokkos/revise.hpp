
#if 0

template<class scalar_t>
void storeEigenvectors(Matrix<scalar_t>* G,
		       Kokkos::VIew<scalar_t**, MemSpace> eigenvectors)
{
  // // Allocate memory for eigenvectors
  // int numRows = G->nrows();
  // eigenvectors = MemoryManager::safe_new<Matrix<scalar_t>>(numRows,numEvecs);
  // Copy appropriate eigenvectors
  int nToCopy = numRows*numEvecs;
  const int ONE = 1;
  Tucker::copy(&nToCopy, G->data(), &ONE, eigenvectors->data(), &ONE);
}

#endif
