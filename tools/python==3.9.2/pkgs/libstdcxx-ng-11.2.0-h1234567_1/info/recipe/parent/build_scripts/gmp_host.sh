#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

GMP_HOST=$HOST
OSX_CONFIG=
case "${HOST}" in
    *darwin*)
        OSX_CONFIG="--with-pic"
        ;;
    *power*)
        GMP_HOST="power8-pc-linux-gnu"
        ;;
esac

rm -rf "${WDIR}/build/gmp-host"
mkdir "${WDIR}/build/gmp-host"
pushd "${WDIR}/build/gmp-host"

    CC="${HOST}-gcc"                          \
    CFLAGS="-pipe ${HOST_CFLAG} -fexceptions" \
    LDFLAGS="${HOST_LDFLAG}"                  \
    bash "${WDIR}/gmp/configure"              \
        --build=${HOST}                       \
        --host=${GMP_HOST}                    \
        --prefix="${WDIR}/buildtools"         \
        --enable-fft                          \
        --enable-cxx                          \
        --disable-shared                      \
        --enable-static ${OSX_CONFIG}

    echo "Building gmp ..."
    make

    echo "Checking gmp ..."
    make -s check

    echo "Installing gmp ..."
    make install

popd

# clean up ...
rm -rf "${WDIR}/build/gmp-host"

