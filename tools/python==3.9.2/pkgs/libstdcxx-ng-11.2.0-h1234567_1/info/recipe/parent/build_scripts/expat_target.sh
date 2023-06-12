#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/expat-target"
mkdir "${WDIR}/build/expat-target"
pushd "${WDIR}/build/expat-target"

    CFLAGS="${ARCH_CFLAG}"                    \
    LDFLAGS="${TARGET_LDFLAG} ${ARCH_LDFLAG}" \
    bash "${WDIR}/expat/configure"            \
        --build=${HOST}                       \
        --host=${CFG_TARGET}                  \
        --prefix="/usr"                       \
        --enable-static                       \
        --enable-shared                       \
        --without-docbook

    echo "Building expat ..."
    make

    echo "Installing expat ..."
    make install DESTDIR="${WDIR}/gcc_built/${CFG_TARGET}/sysroot"

popd

# clean up ...
rm -rf "${WDIR}/build/expat-target"

