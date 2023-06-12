#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/strace-target"
mkdir -p "${WDIR}/build/strace-target"
pushd "${WDIR}/build/strace-target"

    CC="${CFG_TARGET}-gcc"                            \
    CFLAGS="${ARCH_CFLAG}"                           \
    LDFLAGS="${TARGET_LDFLAG} ${ARCH_LDFLAG}"        \
    CPP="${CFG_TARGET}-cpp"                           \
    LD="${CFG_TARGET}-ld"                             \
    bash "${WDIR}/strace/configure"            \
        --build=${HOST}                           \
        --host=${CFG_TARGET}                          \
        --prefix=/usr                                \
        --enable-mpers=check

    echo "Building strace ..."
    make

    echo "Installing strace ..."
    make DESTDIR="${WDIR}/gcc_built/${CFG_TARGET}/debug-root" install

popd

