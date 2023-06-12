set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
declare -a COMMON_MAKE_OPTS=()
COMMON_MAKE_OPTS+=(prefix=${PREFIX} exec_prefix=${PREFIX})

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/gcc-final/

make -C ${CHOST}/libgcc "${COMMON_MAKE_OPTS[@]}" install

# These go into libgcc output
rm -rf ${PREFIX}/${CHOST}/lib
# This is in gcc_impl as it is gcc specific and clang has the same header
rm -rf ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}/include/unwind.h

popd

