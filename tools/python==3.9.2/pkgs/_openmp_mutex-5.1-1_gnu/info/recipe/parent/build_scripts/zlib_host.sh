#!/bin/bash

set -ex

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/zlib-host"
mkdir -p "${WDIR}/build/zlib-host"
pushd "${WDIR}/build/zlib-host"

    CFLAGS="-pipe ${HOST_CFLAG}"      \
    LDFLAGS="${HOST_LDFLAG}"          \
    CHOST="${HOST}"                   \
    bash "${WDIR}/zlib/configure"     \
        --prefix="${WDIR}/buildtools" \
        --static

    echo "Building zlib ..."
    make

    echo "Checking zlib ..."
    make -s test

    echo "Installing zlib ..."
    make install

popd

# clean up ...
rm -rf "${WDIR}/build/zlib-host"

