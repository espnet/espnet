#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

get_min_kernel_config()
{
    version_file="${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/include/linux/version.h"
    if [ ! -f "${version_file}" -o ! -r "${version_file}" ]; then
         echo "The linux version header is unavailable in installed headers"
         exit 1
    fi
    v=$(grep -E LINUX_VERSION_CODE "${version_file}" | cut -d' ' -f 3)
    v1=$(((v>>16)&0xFF))
    v2=$(((v>>8)&0xFF))
    v3=$((v&0xFF))
    echo "--enable-kernel=${v1}.${v2}.${v3}"
}

rm -rf "${WDIR}/build/glibc-core"
mkdir -p "${WDIR}/build/glibc-core"
pushd "${WDIR}/build/glibc-core"

    EXTRA_ADDONS="nptl"
    EXTRA_CFLAGS=
    EXTRA_CONFIG=
    EXTRA_ASFLAG=
    if [ "${CFG_ARCH}" = "x86" ]; then
        EXTRA_CONFIG="--enable-obsolete-rpc"
        EXTRA_CFLAGS="-fcommon -Wno-missing-attributes -Wno-array-bounds -Wno-array-parameter -Wno-stringop-overflow -Wno-maybe-uninitialized"
    elif [ "${CFG_ARCH}" = "powerpc" ]; then
        EXTRA_CFLAGS="-fcommon -Wno-missing-attributes -Wno-array-bounds -Wno-array-parameter -Wno-stringop-overflow -Wno-maybe-uninitialized"
        EXTRA_ADDONS="ports,nptl"
        EXTRA_ASFLAG="-DBROKEN_PPC_8xx_CPU15"
    elif [ "${CFG_ARCH}" = "arm" ]; then
        EXTRA_CFLAGS="-fcommon -Wno-missing-attributes -Wno-array-bounds -Wno-array-parameter -Wno-stringop-overflow -Wno-maybe-uninitialized"
        # building glibc >=2.26
        EXTRA_ADDONS=""
    fi

    touch config.cache

    echo "libc_cv_ssp=no" >>config.cache
    echo "libc_cv_ssp_strong=no" >>config.cache
    echo "libc_cv_forced_unwind=yes" >>config.cache
    echo "libc_cv_c_cleanup=yes" >>config.cache

    printf "\n" > configparms

    echo "ac_cv_path_BASH_SHELL=/bin/bash" >>config.cache

    mkdir -p "${WDIR}/gcc_built/${CFG_TARGET}/sysroot"

    pushd "${WDIR}/gcc_built/bin"
         for t in "${CFG_TARGET}-"*; do
              if [ "${t}" = "${CFG_TARGET}-*" ]; then
                   break
              fi
              _t="${CFG_TARGET}-${t#${CFG_TARGET}-}"
              ln -sfv "${WDIR}/gcc_built/bin/${t}" "${WDIR}/buildtools/bin/${_t}"
         done
    popd

    export LDFLAGS="${TARGET_LDFLAG} ${ARCH_LDFLAG}"

    BUILD_CC="${HOST}-gcc"                                                        \
    CC="${CFG_TARGET}-gcc -g -O2 -U_FORTIFY_SOURCE ${ARCH_CFLAG} ${EXTRA_CFLAGS}" \
    CPPFLAGS="-U_FORTIFY_SOURCE"                                                  \
    CFLAGS="-U_FORTIFY_SOURCE"                                                    \
    AR="${CFG_TARGET}-ar"                                                         \
    RANLIB="${CFG_TARGET}-ranlib" LIBS=""                                         \
    bash "${WDIR}/glibc/configure"                                                \
        --prefix=/usr                                                             \
        --build=${HOST}                                                           \
        --host=${CFG_TARGET}                                                      \
        --cache-file="$(pwd)/config.cache"                                        \
        --without-cvs                                                             \
        --disable-profile                                                         \
        --without-gd                                                              \
        --with-headers="${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/include"      \
        --with-binutils="${WDIR}/gcc_built/bin"                                   \
        --disable-debug --disable-sanity-checks                                   \
        $(get_min_kernel_config)                                                  \
        --with-__thread --with-tls                                                \
        --enable-shared                                                           \
        --enable-add-ons="${EXTRA_ADDONS}"                                        \
        --with-pkgversion="Anaconda glibc"                                        \
        --disable-werror ${EXTRA_CONFIG}

    OSX_CPP=
    OSX_LD=
    case "$HOST" in
        *darwin*)
            OSX_CPP="-I${WDIR}/buildtools/include/"
            OSX_LD="-lintl -liconv"
            ;;
    esac

    echo "Building glibc library ..."

    ASFLAGS="${EXTRA_ASFLAG}"                                                 \
    make default_cflags= CXX= BUILD_CFLAGS="${HOST_CFLAG}"                    \
         BUILD_CPPFLAGS="${OSX_CPP}" BUILD_LDFLAGS="${HOST_LDFLAG} ${OSX_LD}" \
         all

    echo "Installing glibc library ..."

    make default_cflags= CXX= BUILD_CFLAGS="${HOST_CFLAG}"                    \
         BUILD_CPPFLAGS="${OSX_CPP}" BUILD_LDFLAGS="${HOST_LDFLAG} ${OSX_LD}" \
         install_root="${WDIR}/gcc_built/${CFG_TARGET}/sysroot"               \
         install
popd

# save space ... clean up
rm -rf "${WDIR}/build/glibc-core"

