#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/ncurses-build"
mkdir "${WDIR}/build/ncurses-build"
pushd "${WDIR}/build/ncurses-build"

    CFLAGS="${HOST_CFLAG}"                         \
    LDFLAGS="${HOST_LDFLAG}"                       \
    bash "${WDIR}/ncurses/configure"               \
        --build=${HOST}                            \
        --host=${HOST}                             \
        --prefix=""                                \
        --with-install-prefix="${WDIR}/buildtools" \
        --without-debug                            \
        --enable-termcap                           \
        --enable-symlinks                          \
        --without-manpages                         \
        --without-tests                            \
        --without-cxx                              \
        --without-cxx-binding                      \
        --without-ada                              \
        --without-fallbacks

    echo "Building ncurses ..."
    make

    echo "Installing ncurses ..."
    STRIP="${HOST}-strip" make "install.progs"

popd

# clean up ...
rm -rf "${WDIR}/build/ncurses-build"

