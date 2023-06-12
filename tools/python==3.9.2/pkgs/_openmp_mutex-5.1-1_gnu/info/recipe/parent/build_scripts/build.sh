#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

if [ $(id -u) -eq 0 ]; then
    echo "WARNING: You cannot run this script as root and expecting save results"
    # exit 1
fi

n_open_files=$(ulimit -n)
if [ "${n_open_files}" -lt 2048 ]; then
    echo "Number of open files ${n_open_files} may not be sufficient to build the toolchain; increasing to 2048"
    ulimit -n 2048
fi

${RECIPE_DIR}/decompress.sh

mkdir -p "${WDIR}/build"
mkdir -p "${WDIR}/buildtools/bin"

echo "Preparing working directories ..."

rm -rf "${WDIR}/build" "${WDIR}/buildtools" "${WDIR}/gcc_built"

mkdir -p "${WDIR}/gcc_built"
chmod -R u+w "${WDIR}/gcc_built"

mkdir -p "${WDIR}/gcc_built/lib"
ln -s "${WDIR}/gcc_built/lib" "${WDIR}/gcc_built/lib64"

mkdir -p "${WDIR}/gcc_built/${CFG_TARGET}/debug-root/usr/lib"
ln -s "${WDIR}/gcc_built/${CFG_TARGET}/debug-root/usr/lib" "${WDIR}/gcc_built/${CFG_TARGET}/debug-root/usr/lib64"

mkdir -p "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/include"
mkdir -p "${WDIR}/gcc_built/${CFG_TARGET}/lib"
ln -s "${WDIR}/gcc_built/${CFG_TARGET}/lib" "${WDIR}/gcc_built/${CFG_TARGET}/lib64"
mkdir -p "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/lib"
ln -s "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/lib" "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/lib64"
mkdir -p "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/lib"
ln -s "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/lib" "${WDIR}/gcc_built/${CFG_TARGET}/sysroot/usr/lib64"

mkdir -p "${WDIR}/build"
mkdir -p "${WDIR}/buildtools/bin"

echo "Making build system tools available with poisoned name"
for tool in ar as dlltool c++ c++filt cpp cc gcc gcc-ar gcc-nm gcc-ranlib \
            g++ gcj gnatbind gnatmake ld libtool nm objcopy objdump ranlib \
            strip windres; do
    where=$(which "${ORG_HOST}-${tool}" 2>/dev/null || true)
    [ -z "${where}" ] && where=$(which "${tool}" 2>/dev/null || true)

    if [ -n "${where}" ]; then
        printf "#!/bin/bash\nexec '${where}' \"\${@}\"\n" >"${WDIR}/buildtools/bin/${HOST}-${tool}"
        chmod 700 "${WDIR}/buildtools/bin/${HOST}-${tool}"
    fi
done

# for ncurses sake on powerpc and s390x we create a symlink for gcc
pushd "${WDIR}/buildtools/bin"
ln -s "${HOST}-gcc" gcc
ln -s "${HOST}-g++" g++ || true
popd

printf "#!/bin/bash\n$(which makeinfo 2>/dev/null || true) --force \"\${@}\"\ntrue\n" >"${WDIR}/buildtools/bin/makeinfo"
chmod 700 "${WDIR}/buildtools/bin/makeinfo"

echo "Checking that we can run gcc --version and being able to compile program ..."
echo "  ${HOST}-gcc --version ..."
"${HOST}-gcc" --version || (ls compilers/bin && exit 1)
    
printf "int main()\n{\n  return 0;\n}\n" >"${WDIR}/build/test.c"
"${HOST}-gcc" -pipe ${HOST_CFLAG} ${HOST_LDFLAG} "${WDIR}/build/test.c" -o "${WDIR}/build/out"
rm -f "${WDIR}/build/test.c" "${WDIR}/build/out"

echo "Starting the build ..."
bash ${RECIPE_DIR}/build_scripts/ncurses_build.sh 
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/zlib_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/gmp_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/mpfr_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/isl_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/mpc_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/expat_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/ncurses_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/libiconv_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/gettext_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/binutils_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/kernel_headers.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/gcc_core.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/glibc_core.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/gcc_host.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/gmp_target.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/expat_target.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/ncurses_target.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/libelf_target.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/debug.sh
 [ $? -eq 0 ]
bash ${RECIPE_DIR}/build_scripts/final.sh
 [ $? -eq 0 ]

exit 0
