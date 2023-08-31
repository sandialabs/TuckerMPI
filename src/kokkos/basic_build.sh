#!/usr/bin/env bash

set -e

if [[ ( $@ == "--help") ||  $@ == "-h" ]]
then
    echo "Usage: $0 <fullpath-to-your-desired-workdir> <fullpath-to-tucker-repo>"
    exit 0
fi

MYPWD=`pwd`
WORKDIR=$1
TUCKERREPO=$2

if [ -z "$CXX" ]; then
    echo "CXX must be a valid C++17 compiler. If you want an mpi build, you must set it to point to your MPI C++ wrapper (mpic++). Terminating."
    exit 1
fi

if [ -z "${WORKDIR}" ]; then
    echo "WORKDIR cannot be empty. Terminating."
    exit 2
fi

if [ ! -d ${WORKDIR} ]; then
    mkdir -p ${WORKDIR}
    echo "${WORKDIR} does not exist, creating it."
fi


# --- nothing to addapt below --- #

function build_openblas(){
    openblasVers=0.3.22

    if [ ! -f v${openblasVers}.tar.gz ]; then
	wget https://github.com/xianyi/OpenBLAS/archive/v${openblasVers}.tar.gz
    fi

    if [ ! -d OpenBLAS-${openblasVers} ]; then
	tar zxf v${openblasVers}.tar.gz
    fi

    cd OpenBLAS-${openblasVers}

    BFName="${PWD}/../build.txt"
    IFName="${PWD}/../install.txt"

    echo "Building OpenBLAS"
    make BINARY=64 HOSTCC=$CC > ${BFName} 2>&1
    echo "Build output written to ${BFName}"
    if grep -q "OpenBLAS build complete" "${BFName}"; then
	echo "OpenBLAS built successfull"
    else
	echo "OpenBLAS built unsuccessfull"
	exit 44
    fi

    echo "Installing OpenBLAS"
    make PREFIX=${BLASROOT} install > ${IFName} 2>&1
    echo "Install output written to ${IFName}"
}

function buildKokkos(){
    if [[ -d ${KOKKOSPFX} ]]; then
	echo "Kokkos already installed, skipping"
    else

	if [[ ! -d ${KOKKOS_BUILD_DIR} ]]; then
	    mkdir -p ${KOKKOS_BUILD_DIR}
	else
	    rm -rf ${KOKKOS_BUILD_DIR}/*
	fi

	cd ${KOKKOS_BUILD_DIR}
	cmake -DCMAKE_CXX_COMPILER=${CXX} \
	      -DCMAKE_BUILD_TYPE="Release" \
	      -DCMAKE_INSTALL_PREFIX=${KOKKOSPFX} \
	      -DKokkos_ENABLE_TESTS=Off \
	      -DKokkos_ENABLE_SERIAL=Off \
	      -DKokkos_ENABLE_OPENMP=On \
	      -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=Off \
	      ${KOKKOS_SRC}
	make -j8
	make install
	cd ..
    fi
}

function buildKokkosKernels(){
    echo "Kernels using the KokkosPFX=${KOKKOSPFX}"

    if [[ -d ${KOKKOSKERPFX} ]]; then
	echo "Kokkos-kernels already installed, skipping"
    else
	if [[ ! -d ${KOKKOSKER_BUILD_DIR} ]]; then
	    mkdir -p ${KOKKOSKER_BUILD_DIR}
	else
	    rm -rf ${KOKKOSKER_BUILD_DIR}/*
	fi

	cd ${KOKKOSKER_BUILD_DIR}

	cmake \
	    -DCMAKE_VERBOSE_MAKEFILE=On \
	    -DCMAKE_CXX_COMPILER=${CXX} \
	    -DCMAKE_BUILD_TYPE="Release" \
	    -DCMAKE_INSTALL_PREFIX=${KOKKOSKERPFX} \
	    \
	    -DBLAS_LIBRARIES=openblas \
	    -DLAPACK_LIBRARIES=openblas \
	    -DKokkosKernels_BLAS_ROOT=${BLASROOT} \
	    -DKokkosKernels_LAPACK_ROOT=${BLASROOT} \
	    -DKokkosKernels_ENABLE_TPL_LAPACK=On \
	    -DKokkosKernels_ENABLE_TPL_BLAS=On \
	    \
	    -DKokkosKernels_INST_DOUBLE=On \
	    -DKokkosKernels_INST_LAYOUTRIGHT=On \
	    -DKokkosKernels_INST_LAYOUTLEFT=On \
	    -DKokkosKernels_INST_ORDINAL_INT=Off \
	    -DKokkosKernels_INST_ORDINAL_INT64_T=On \
	    -DKokkosKernels_INST_OFFSET_INT=Off \
	    -DKokkosKernels_INST_OFFSET_SIZE_T=On \
	    \
	    -DKokkosKernels_ENABLE_TESTS=Off \
	    -DKokkos_ROOT=${KOKKOSPFX} \
	    ${KOKKOS_KER_SRC}

	make -j8
	make install
	cd ..
    fi
}

function cloneKokkoses
{
    if [ ! -f kokkos-${kokkosversion}.tar.gz ]; then
	wget https://github.com/kokkos/kokkos/archive/refs/tags/${kokkosversion}.tar.gz
	mv ${kokkosversion}.tar.gz kokkos-${kokkosversion}.tar.gz
    fi
    if [ ! -d kokkos-${kokkosversion} ]; then
	tar zxf kokkos-${kokkosversion}.tar.gz
    fi

    if [ ! -f kokkos-kernels-${kokkosversion}.tar.gz ]; then
	wget https://github.com/kokkos/kokkos-kernels/archive/refs/tags/${kokkosversion}.tar.gz
	mv ${kokkosversion}.tar.gz kokkos-kernels-${kokkosversion}.tar.gz
    fi
    if [ ! -d kokkos-kernels-${kokkosversion} ]; then
	tar zxf kokkos-kernels-${kokkosversion}.tar.gz
    fi
}

cd $WORKDIR

#
# BLAS/LAPACK
#
export BLASROOT=${WORKDIR}/openblas/install
if [[ -d ${BLASROOT} ]]; then
    echo "openblas already found in ${BLASROOT}, skipping"
else
    [[ ! -d ${WORKDIR}/openblas ]] && mkdir ${WORKDIR}/openblas
    cd ${WORKDIR}/openblas
    build_openblas
    cd ${MYPWD}
fi

#
# Kokkos
#
export kokkosversion=4.0.00
export KOKKOSESWORKDIR=${WORKDIR}/kokkoses
[[ ! -d ${KOKKOSESWORKDIR} ]] && mkdir ${KOKKOSESWORKDIR}

export KOKKOS_SRC=${KOKKOSESWORKDIR}/kokkos-${kokkosversion}
export KOKKOS_KER_SRC=${KOKKOSESWORKDIR}/kokkos-kernels-${kokkosversion}
export KOKKOS_BUILD_DIR=${KOKKOSESWORKDIR}/kokkos-build
export KOKKOSKER_BUILD_DIR=${KOKKOSESWORKDIR}/kokkos-kernels-build
export KOKKOSPFX=${KOKKOSESWORKDIR}/kokkos-install
export KOKKOSKERPFX=${KOKKOSESWORKDIR}/kokkos-kernels-install

cd ${KOKKOSESWORKDIR}
cloneKokkoses
buildKokkos openmp
buildKokkosKernels
cd ${MYPWD}


export TUCKERWORKDIR=${WORKDIR}/build-tucker
if [[ ! -d ${TUCKERWORKDIR} ]]; then
    mkdir ${TUCKERWORKDIR} && cd ${TUCKERWORKDIR}

    # sometimes kernels installation gets a lib64 instead of lib
    if [[ -d ${KOKKOSKERPFX}/lib ]]; then
	echo "GIGI"
	export KKERDIR=${KOKKOSKERPFX}/lib/cmake/KokkosKernels
    else
	export KKERDIR=${KOKKOSKERPFX}/lib64/cmake/KokkosKernels
    fi

    if [ -z "${MPI_HOME}" ]; then
	echo " "
	echo "!!!! IMPORTANT !!!!"
	echo "The env var MPI_HOME is empty, so I am building Tucker for onnode only".
	echo "If you want an MPI build, you must set MPI_HOME to point to the base directory of your MPI installation"
	echo " "

	cmake -DCMAKE_BUILD_TYPE=Release \
	      -DCMAKE_CXX_STANDARD=17 \
	      -DKokkosKernels_DIR=${KKERDIR} \
	      -DCMAKE_CXX_COMPILER=${CXX} \
	      -DTUCKER_ENABLE_KOKKOS=ON \
	      ${TUCKERREPO}/src
	make -j6
    else

	cmake -DCMAKE_BUILD_TYPE=Release \
	      -DMPI_HOME=${MPI_HOME} \
	      -DCMAKE_CXX_STANDARD=17 \
	      -DKokkosKernels_DIR=${KKERDIR} \
	      -DCMAKE_CXX_COMPILER=${CXX} \
	      -DTUCKER_ENABLE_KOKKOS=ON -DTUCKER_ENABLE_MPI=ON \
	      ${TUCKERREPO}/src
	make -j6
    fi

    cd ${MYPWD}
else
    echo "Tucker build already found in ${TUCKERWORKDIR}, skipping"
fi
