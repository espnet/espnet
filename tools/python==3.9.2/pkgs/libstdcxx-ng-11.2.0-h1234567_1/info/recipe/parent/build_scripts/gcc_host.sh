#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

rm -rf "${WDIR}/build/gcc-final"
mkdir "${WDIR}/build/gcc-final"
pushd "${WDIR}/build/gcc-final"

    ARCH_CONFIG=
    if [ "${CFG_ARCH}" = "x86" ]; then
        ARCH_CONFIG="--enable-libmpx"
    elif [ "${CFG_ARCH}" = "arm" ]; then
        ARCH_CONFIG="--enable-gold"
    fi

    CLANG_CFLAG=
    if ${HOST}-gcc --version 2>&1 | grep clang; then
        CLANG_CFLAG="-fbracket-depth=512"
    fi

    CC_FOR_BUILD="${HOST}-gcc"                                        \
    CFLAGS="${HOST_CFLAG} ${CLANG_CFLAG}"                             \
    CFLAGS_FOR_BUILD="${HOST_CFLAG} ${CLANG_CFLAG}"                   \
    CXXFLAGS="${HOST_CFLAG} ${CLANG_CFLAG}"                           \
    CXXFLAGS_FOR_BUILD="${HOST_CFLAG} ${CLANG_CFLAG}"                 \
    LDFLAGS="${HOST_LDFLAG} -lstdc++ -lm"                             \
    CFLAGS_FOR_TARGET="-g -O2 ${ARCH_CFLAG}"                          \
    CXXFLAGS_FOR_TARGET="${ARCH_CFLAG}"                               \
    LDFLAGS_FOR_TARGET="${TARGET_LDFLAG} ${ARCH_LDFLAG}"              \
    bash "${WDIR}/gcc/configure"                                      \
        --build=${HOST}                                               \
        --host=${HOST}                                                \
        --target=${CFG_TARGET}                                        \
        --prefix="${WDIR}/gcc_built"                                  \
        --exec_prefix="${WDIR}/gcc_built"                             \
        --with-sysroot="${WDIR}/gcc_built/${CFG_TARGET}/sysroot"      \
        --with-local-prefix="${WDIR}/gcc_built/${CFG_TARGET}/sysroot" \
        --enable-long-long                                            \
        --disable-multilib                                            \
        --disable-nls                                                 \
        --with-gmp=${WDIR}/buildtools                                 \
        --with-mpfr=${WDIR}/buildtools                                \
        --with-mpc=${WDIR}/buildtools                                 \
        --with-isl=${WDIR}/buildtools                                 \
        --without-zstd                                                \
        --enable-languages="c,c++,fortran,objc,obj-c++"               \
        --enable-__cxa_atexit                                         \
        --disable-libmudflap                                          \
        --enable-libgomp                                              \
        --enable-libquadmath                                          \
        --enable-libquadmath-support                                  \
        --enable-libsanitizer                                         \
        --disable-libstdcxx-verbose                                   \
        --enable-lto --enable-libcc1                                  \
        --enable-threads=posix                                        \
        --enable-plugin                                               \
        --with-pkgversion="Anaconda gcc"                              \
        "${ARCH_CONFIG}"

    echo "Building final gcc compiler  ..."
    make all

    echo "Installing final gcc compiler ..."
    make install-strip

    make install-target-libobjc

    pushd "${WDIR}/gcc_built"
        find . -type f -name "*.la" -exec rm {} \;
    popd

    if [ -f "${WDIR}/gcc_built/bin/${CFG_TARGET}-gcc" ]; then
        ln -sfv "${WDIR}/gcc_built/bin/${CFG_TARGET}-gcc" "${WDIR}/gcc_built/bin/${CFG_TARGET}-cc"
    fi

    multi_root=$( "${WDIR}/gcc_built/bin/${CFG_TARGET}-gcc" -print-sysroot )
    if [ ! -e "${multi_root}/lib" ]; then
        ln -sfv lib "${multi_root}/lib"
    fi
    if [ ! -e "${multi_root}/usr/lib" ]; then
        ln -sfv lib "${multi_root}/usr/lib"
    fi
    echo $multi_root
    if [ -d "${WDIR}/gcc_built/lib/bfd-plugins" ]; then
        gcc_version=$(cat "${WDIR}/gcc/gcc/BASE-VER" )
        ln -sfv "../../libexec/gcc/${CFG_TARGET}/${gcc_version}/liblto_plugin.so" "${WDIR}/gcc_built/lib/bfd-plugins/liblto_plugin.so"
    fi

popd

multi_root=$( "${CFG_TARGET}-gcc" -print-sysroot )

canon_root=$( cd "${multi_root}" && pwd -P )
canon_prefix=$( cd "${WDIR}/gcc_built" && pwd -P )

gcc_dir="${WDIR}/gcc_built/${CFG_TARGET}/lib"
if [ ! -d "${gcc_dir}" ]; then
    return
fi
dst_dir="${canon_root}/lib"
rel=$( echo "${gcc_dir#${WDIR}/gcc_built/}" | sed 's#[^/]\{1,\}#..#g' )

ls "${gcc_dir}" | while read f; do
    case "${f}" in
        *.ld)
            continue
            ;;
    esac
    if [ -f "${gcc_dir}/${f}" ]; then
        mkdir -p "${dst_dir}"
        mv "${gcc_dir}/${f}" "${dst_dir}/${f}"
        ln -sf "${rel}/${dst_dir#${canon_prefix}/}/${f}" "${gcc_dir}/${f}"
        fi
done

# require working host "gcc" tool for successful configuration of ncurses, strace, ...
pushd "gcc_built/bin"
ln -s ${CFG_TARGET}-gcc gcc
popd

