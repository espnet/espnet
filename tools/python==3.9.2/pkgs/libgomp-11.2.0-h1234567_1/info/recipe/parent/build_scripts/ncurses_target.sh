#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/ncurses-target"
mkdir "${WDIR}/build/ncurses-target"
pushd "${WDIR}/build/ncurses-target"

    CFLAGS="${ARCH_CFLAG}"                                              \
    LDFLAGS="${TARGET_LDFLAG}"                                          \
    bash "${WDIR}/ncurses/configure"                                    \
        --build=${HOST}                                                 \
        --host=${CFG_TARGET}                                            \
        --prefix="/usr"                                                 \
        --with-install-prefix="${WDIR}/gcc_built/${CFG_TARGET}/sysroot" \
        --without-debug                                                 \
        --enable-termcap                                                \
        --with-shared                                                   \
        --without-sysmouse

    echo "Building ncurses ..."
    make

    echo "Installing ncurses ..."
    STRIP="${CFG_TARGET}-strip" make install

popd

# clean up ...
rm -rf "${WDIR}/build/ncurses-target"

