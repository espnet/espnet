#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

echo "Finalizing the toolchain's directories ..."

STRIP_ARG="--strip-unneeded -v"
case "$HOST" in
    *darwin*)
       STRIP_ARG=""
       ;;
esac

echo "Stripping all executables in prefix ..."
pushd "${WDIR}/gcc_built"

    if [ -f "${CFG_TARGET}/debug-root/usr/bin/gdbserver" ]; then
        "${CFG_TARGET}-strip" ${STRIP_ARG} "${CFG_TARGET}/debug-root/usr/bin/gdbserver"
    fi
    gcc_version=$(cat "${WDIR}/gcc/gcc/BASE-VER")
    for _t in "bin/${CFG_TARGET}-"*                                      \
              "${CFG_TARGET}/bin/"*                                      \
              "libexec/gcc/${CFG_TARGET}/${gcc_version}/"*               \
              "libexec/gcc/${CFG_TARGET}/${gcc_version}/install-tools/"* \
        ; do
            _type="$( file "${_t}" |cut -d ' ' -f 2- )"
        case "${_type}" in
            *script*executable*)
                ;;
            *executable*|*shared*object*)
                ${HOST}-strip ${STRIP_ARG} "${_t}"
                ;;
        esac
    done
popd

rm -rf  "${WDIR}/gcc_built/"{,usr/}{,share/}{man,info}
rm -rf  "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/"{,usr/}{,share/}{man,info}
rm -rf  "${WDIR}/gcc_built/${CFG_TARGET}/debug-root/"{,usr/}{,share/}{man,info}

for licfile in $( find "${WDIR}" -follow -type f -a \( -name "COPYING*" -o -name "LICENSE*" \) ); do
    dstdir="${licfile%/*}"
    dstdir="${WDIR}/gcc_built/share/licenses${dstdir#${WDIR}}"
    mkdir -p "${dstdir}"
    cp -av "${licfile}" "${dstdir}/"
done

shopt -u nullglob
