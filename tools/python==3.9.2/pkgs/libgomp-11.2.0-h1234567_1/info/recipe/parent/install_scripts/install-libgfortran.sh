set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

mkdir -p ${PREFIX}/lib/
rm -f ${PREFIX}/lib/libgfortran* || true

cp -f --no-dereference ${SRC_DIR}/build/gcc-final/${CHOST}/libgfortran/.libs/libgfortran*.so* ${PREFIX}/lib/

# patchelf --set-rpath '$ORIGIN' ${PREFIX}/lib/libgfortran*.so*

# Install Runtime Library Exception
install -Dm644 $SRC_DIR/gcc/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/libgfortran/RUNTIME.LIBRARY.EXCEPTION
