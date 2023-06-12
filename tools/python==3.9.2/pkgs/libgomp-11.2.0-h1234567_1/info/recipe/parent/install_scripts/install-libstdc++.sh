set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
declare -a COMMON_MAKE_OPTS=()
COMMON_MAKE_OPTS+=(prefix=${PREFIX} exec_prefix=${PREFIX})

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/gcc-final/

  make -C ${CHOST}/libstdc++-v3/src "${COMMON_MAKE_OPTS[@]}" install-toolexeclibLTLIBRARIES
  make -C ${CHOST}/libstdc++-v3/po "${COMMON_MAKE_OPTS[@]}" install

popd

mkdir -p ${PREFIX}/lib
mv ${PREFIX}/${CHOST}/lib64/* ${PREFIX}/lib

# patchelf --set-rpath '$ORIGIN' ${PREFIX}/lib/libstdc++.so

# no static libs
find ${PREFIX}/lib -name "*\.a" -exec rm -rf {} \;
# no libtool files
find ${PREFIX}/lib -name "*\.la" -exec rm -rf {} \;

# Install Runtime Library Exception
install -Dm644 ${SRC_DIR}/gcc/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/libstdc++/RUNTIME.LIBRARY.EXCEPTION
