#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/isl-host"
mkdir "${WDIR}/build/isl-host"
pushd "${WDIR}/build/isl-host"

    CFLAGS="-pipe ${HOST_CFLAG}"               \
    CXXFLAGS="-pipe ${HOST_CFLAG}"             \
    LDFLAGS="${HOST_LDFLAG}"                   \
    bash "${WDIR}/isl/configure"               \
        --build=${HOST}                        \
        --host=${HOST}                         \
        --prefix="${WDIR}/buildtools"          \
        "${extra_config[@]}"                   \
        --disable-shared                       \
        --enable-static                        \
        --with-gmp=system                      \
        --with-gmp-prefix="${WDIR}/buildtools" \
        --with-clang=no

    echo "Building ISL ..."
    make

    echo "Checking ISL ..."
    make -s check

    echo "Installing ISL ..."
    make install

popd

# clean up ...
rm -rf "${WDIR}/build/isl-host"

