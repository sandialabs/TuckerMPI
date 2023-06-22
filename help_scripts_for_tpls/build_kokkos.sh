#!/usr/bin/env bash

set -e

MYPWD=`pwd`

BLASROOT=${PWD}/openblas/install
DESTDIR=${PWD}/kokkoses
PFXDIR=${DESTDIR}
kokkosversion=4.0.00

# --- nothing to addapt below --- #

function buildKokkos(){
    [[ ! -d ${KOKKOS_BUILD_DIR} ]] && mkdir -p ${KOKKOS_BUILD_DIR}
    cd ${KOKKOS_BUILD_DIR}

	cmake -DCMAKE_CXX_COMPILER=${CXX} \
	      -DCMAKE_BUILD_TYPE="Release" \
	      -DCMAKE_INSTALL_PREFIX=${KOKKOSPFX} \
	      -DKokkos_ENABLE_TESTS=Off \
	      -DKokkos_ENABLE_SERIAL=On \
	      -DKokkos_ENABLE_OPENMP=Off \
	      -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=Off \
	      ${KOKKOS_SRC}
    make -j16
    make install
    cd ..
}

function buildKokkosKernels(){
    echo "Kernels using the KokkosPFX=${KOKKOSPFX}"

    [[ ! -d ${KOKKOSKER_BUILD_DIR} ]] && mkdir -p ${KOKKOSKER_BUILD_DIR}
    cd ${KOKKOSKER_BUILD_DIR} && rm -rf CMakeCache* src/*

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

    make -j16
    make install
    cd ..
}

function doKokkos
{
    cd ${DESTDIR}

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

    # sources for kokkos
    KOKKOS_SRC=${DESTDIR}/kokkos-${kokkosversion}
    KOKKOS_KER_SRC=${DESTDIR}/kokkos-kernels-${kokkosversion}
    # build dirs
    KOKKOS_BUILD_DIR=${DESTDIR}/kokkos-build
    KOKKOSKER_BUILD_DIR=${DESTDIR}/kokkos-kernels-build
    # prefixes
    KOKKOSPFX=${PFXDIR}/kokkos-install
    KOKKOSKERPFX=${PFXDIR}/kokkos-kernels-install

    buildKokkos ${kokkosbackend}
    buildKokkosKernels ${kokkosbackend}
}


[[ ! -d ${DESTDIR} ]] && mkdir -p ${DESTDIR}
cd ${DESTDIR}

doKokkos

cd ${MYPWD}
