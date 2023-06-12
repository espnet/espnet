set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
declare -a COMMON_MAKE_OPTS=()
COMMON_MAKE_OPTS+=(prefix=${PREFIX} exec_prefix=${PREFIX})

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/gcc-final/

  make -C ${CHOST}/libgcc "${COMMON_MAKE_OPTS[@]}" install-shared

  # TODO :: Also do this for libgfortran (and libstdc++ too probably?)
  sed -i.bak 's/.*cannot install.*/func_warning "Ignoring libtool error about cannot install to a directory not ending in"/' \
             ${CHOST}/libsanitizer/libtool
  for lib in libatomic libgomp libquadmath libitm libvtv libsanitizer/{a,l,ub,t}san; do
    # TODO :: Also do this for libgfortran (and libstdc++ too probably?)
    if [[ -f ${CHOST}/${lib}/libtool ]]; then
      sed -i.bak 's/.*cannot install.*/func_warning "Ignoring libtool error about cannot install to a directory not ending in"/' \
                 ${CHOST}/${lib}/libtool
    fi
    if [[ -d ${CHOST}/${lib} ]]; then
      make -C ${CHOST}/${lib} "${COMMON_MAKE_OPTS[@]}" install-toolexeclibLTLIBRARIES
      make -C ${CHOST}/${lib} "${COMMON_MAKE_OPTS[@]}" install-nodist_fincludeHEADERS || true
    fi
  done

  for lib in libgomp libquadmath; do
    if [[ -d ${CHOST}/${lib} ]]; then
      make -C ${CHOST}/${lib} "${COMMON_MAKE_OPTS[@]}" install-info
    fi
  done

popd

mkdir -p ${PREFIX}/lib

# no static libs
find ${PREFIX}/${CHOST}/lib64 -name "*\.a" -exec rm -rf {} \;
# no libtool files
find ${PREFIX}/${CHOST}/lib64 -name "*\.la" -exec rm -rf {} \;

if [[ "${PKG_NAME}" != gcc_impl* ]]; then
  mv ${PREFIX}/${CHOST}/lib64/* ${PREFIX}/lib
  # clean up empty folder
  rm -rf ${PREFIX}/lib/gcc

  # Install Runtime Library Exception
  install -Dm644 ${SRC_DIR}/gcc/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/gcc-libs/RUNTIME.LIBRARY.EXCEPTION
fi
