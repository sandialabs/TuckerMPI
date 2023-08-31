# Requirements

C++-17 compiler, CMake, Kokkos, MPI, BLAS/LAPACK

Versions known to work: cmake [3.16], gcc [9.5.0, 11.3], kokkos [4.0], openblas [0.3.10, 0.3.22], openmpi [1.10.2]

Clone the Gitlab repository: https://gitlab.com/nga-tucker/TuckerMPI, **``nga-develop`` branch.** <br>

-----

## On-node configure only using Kokkos

```bash
export CC=<fullpath-to-c++-compiler>
export CXX=<fullpath-to-c-compiler>
export FC=<fullpath-to-fortran-compiler>

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DKokkosKernels_DIR=<fullpath-to-installation-dir>/lib/cmake/KokkosKernels \
  -DTUCKER_ENABLE_KOKKOS=ON \
  <fullpath-to-TuckerMPI-repo>/src
```

## Configure for MPI + Kokkos

```bash
export CC=<fullpath-to-mpicc-compiler>
export CXX=<fullpath-to-mpic++-compiler>
export FC=<fullpath-to-mpifortran-compiler>

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DMPI_HOME=<fullpath-to-install-directory-of-mpi>
  -DKokkosKernels_DIR=<fullpath-to-installation-dir>/lib/cmake/KokkosKernels \
  -DTUCKER_ENABLE_KOKKOS=ON \
  -DTUCKER_ENABLE_MPI=ON \
  <fullpath-to-TuckerMPI-repo>/src
```

-----

## Step-by-step

Use the build bash script provided inside `src/kokkos/basic_build.sh`.
This is hardwired to use Kokkos with OpenMP backend.

Example usage:

```bash
export CXX=<fullpath-to-mpic++-compiler>
export MPI_HOME=<fullpath-to-homedir-of-base-directory-of-your-MPI-installation"
export MYWORKDIR=/home/mytuckertest
bash ${PWD}/TuckerMPI/src/kokkos/basic_build.sh ${MYWORKDIR} $PWD/TuckerMPI
```


# Verifying tests

After building, run `ctest` from within the build directory
