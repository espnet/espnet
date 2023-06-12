set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
declare -a COMMON_MAKE_OPTS=()
COMMON_MAKE_OPTS+=(prefix=${PREFIX} exec_prefix=${PREFIX})

_libdir=libexec/gcc/${CHOST}/${PKG_VERSION}
# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/gcc-final/

make -C gcc "${COMMON_MAKE_OPTS[@]}" c++.install-common

# How it used to be:
# install -m755 -t ${PREFIX}/bin/ gcc/{cc1plus,lto1}
for file in cc1plus; do
  if [[ -f gcc/${file} ]]; then
    install -c gcc/${file} ${PREFIX}/${_libdir}/${file}
  fi
done

# Following 3 are in libstdcxx-devel
#make -C $CHOST/libstdc++-v3/src "${COMMON_MAKE_OPTS[@]}" install
#make -C $CHOST/libstdc++-v3/include "${COMMON_MAKE_OPTS[@]}" install
#make -C $CHOST/libstdc++-v3/libsupc++ "${COMMON_MAKE_OPTS[@]}" install
make -C $CHOST/libstdc++-v3/python "${COMMON_MAKE_OPTS[@]}" install

# Probably don't want to do this for cross-compilers
# mkdir -p ${PREFIX}/share/gdb/auto-load/usr/lib/
# cp ${SRC_DIR}/gcc_built/${CHOST}/sysroot/lib/libstdc++.so.6.*-gdb.py ${PREFIX}/share/gdb/auto-load/usr/lib/

make -C libcpp "${COMMON_MAKE_OPTS[@]}" install

popd

set +x
# Strip executables, we may want to install to a different prefix
# and strip in there so that we do not change files that are not
# part of this package.
pushd ${PREFIX}
  _files=$(find . -type f)
  for _file in ${_files}; do
    _type="$( file "${_file}" | cut -d ' ' -f 2- )"
    case "${_type}" in
      *script*executable*)
      ;;
      *executable*)
        ${SRC_DIR}/gcc_built/bin/${CHOST}-strip --strip-unneeded -v "${_file}" || :
      ;;
    esac
  done
popd

source ${RECIPE_DIR}/install_scripts/make_tool_links.sh
