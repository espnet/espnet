#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/gdb-server"
mkdir -p "${WDIR}/build/gdb-server"
pushd "${WDIR}/build/gdb-server"

    export ac_cv_func_strncmp_works=yes

    export CONFIG_SHELL="/bin/bash"
    export SHELL="/bin/bash"

    CC_FOR_BUILD="${HOST}-gcc"                                             \
    CPP="${CFG_TARGET}-cpp"                                                \
    CC="${CFG_TARGET}-gcc"                                                 \
    CXX="${CFG_TARGET}-g++"                                                \
    LD="${CFG_TARGET}-ld"                                                  \
    CFLAGS="${ARCH_CFLAG}"                                                 \
    CXXFLAGS="${ARCH_CFLAG}"                                               \
    LDFLAGS="${TARGET_LDFLAG} ${ARCH_LDFLAG}"                              \
    bash "${WDIR}/gdb/configure"                                           \
        --build=${HOST}                                                    \
        --host=${CFG_TARGET}                                               \
        --target=${CFG_TARGET}                                             \
        --prefix="/usr" --disable-multilib                                 \
        --with-build-sysroot="${WDIR}/gcc_built/${CFG_TARGET}/sysroot"     \
        --includedir="${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/include" \
        --without-uiout                                                    \
        --disable-gdb                                                      \
        --disable-gdbtk                                                    \
        --without-x                                                        \
        --disable-sim                                                      \
        --without-included-gettext                                         \
        --without-develop                                                  \
        --sysconfdir=/etc                                                  \
        --localstatedir=/var                                               \
        --disable-werror                                                   \
        --enable-64bit-bfd                                                 \
        --enable-deterministic-archives                                    \
        --program-prefix=                                                  \
        --enable-gdbserver                                                 \
        --disable-inprocess-agent                                          \
        --disable-binutils                                                 \
        --disable-ld                                                       \
        --disable-gas                                                      \
        --enable-threads                                                   \
        --disable-nls

    echo "Building gdb ..."
    make

    echo "Installing gdb ..."
    make DESTDIR="${WDIR}/gcc_built/${CFG_TARGET}/debug-root" install

    unset ac_cv_func_strncmp_works

popd

