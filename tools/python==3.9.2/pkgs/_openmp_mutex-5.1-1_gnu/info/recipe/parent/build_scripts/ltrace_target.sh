#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/ltrace-target"
mkdir -p "${WDIR}/build/ltrace-target"
pushd "${WDIR}/build/ltrace-target"
    cp -r "${WDIR}"/ltrace/* .

    EXTRA_CONFIG=
    if [ "${CFG_ARCH}" = "arm" ]; then
        cp $BUILD_PREFIX/share/libtool/build-aux/config.* config/autoconf/.
        EXTRA_CONFIG="--disable-werror"
    fi
    # fixes a quirk in makefile, which uses cpu part of triplet
    # to determine subfolder name ...
    pushd "sysdeps/linux-gnu"
    ln -s ppc powerpc64le
    popd

    CONFIG_SHELL="/bin/bash"   \
    LDFLAGS="${TARGET_LDFLAG}" \
    bash ./configure           \
        --build=${HOST}        \
        --host=${CFG_TARGET}   \
        --prefix=/usr          \
        ${EXTRA_CONFIG}        \
        --with-gnu-ld

    echo "Building ltrace ..."
    make

    echo "Installing ltrace ..."
    make DESTDIR="${WDIR}/gcc_built/${CFG_TARGET}/debug-root" install

popd

