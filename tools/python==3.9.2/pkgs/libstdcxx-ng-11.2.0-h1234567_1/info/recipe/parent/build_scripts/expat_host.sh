#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/expat-host"
mkdir "${WDIR}/build/expat-host"
pushd "${WDIR}/build/expat-host"

    CFLAGS="-pipe ${HOST_CFLAG}"      \
    LDFLAGS="${HOST_LDFLAG}"          \
    bash "${WDIR}/expat/configure"    \
        --build=${HOST}               \
        --host=${HOST}                \
        --prefix="${WDIR}/buildtools" \
        --enable-static               \
        --disable-shared              \
        --without-docbook

    echo "Building expat ..."
    make

    echo "Installing expat ..."
    make install

popd

# clean up ...
rm -rf "${WDIR}/build/expat-host"

