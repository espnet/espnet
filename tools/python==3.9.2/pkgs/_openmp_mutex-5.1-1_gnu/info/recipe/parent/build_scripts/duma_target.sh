#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/duma-target"
mkdir -p "${WDIR}/build/duma-target"
pushd "${WDIR}/build/duma-target"
    cp -r "${WDIR}"/duma/* .

    echo "Building duma ..."
    make DUMA_CPP=1 DUMASO= OS="linux"                          \
        CC="${CFG_TARGET}-gcc ${ARCH_CFLAG}"                    \
        CXX="${CFG_TARGET}-g++ ${ARCH_CFLAG}"                   \
        HOSTCC="${HOST}-gcc"                                    \
        RANLIB="${CFG_TARGET}-ranlib"                           \
        LDFLAGS="${TARGET_LDFLAG}"                              \
        prefix="${WDIR}/gcc_built/${CFG_TARGET}/debug-root/usr" \
        all

    echo "Installing duma ..."
    make DUMA_CPP=1 DUMASO= OS="linux"                          \
        CC="${CFG_TARGET}-gcc ${ARCH_CFLAG}"                    \
        CXX="${CFG_TARGET}-g++ ${ARCH_CFLAG}"                   \
        HOSTCC="${HOST}-gcc"                                    \
        RANLIB="${CFG_TARGET}-ranlib"                           \
        LDFLAGS="${TARGET_LDFLAG}"                              \
        prefix="${WDIR}/gcc_built/${CFG_TARGET}/debug-root/usr" \
        install

popd

