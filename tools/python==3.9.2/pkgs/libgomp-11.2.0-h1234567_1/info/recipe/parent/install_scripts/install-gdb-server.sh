set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)

# libtool wants to use ranlib that is here
# for psuedo-cross, depending on the build and host,
# build-gdb-native could be used (it will have been built
# with the newer GCC instead of the system one).
pushd ${SRC_DIR}/build/gdb-server
  export PATH=${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:$PATH
  make prefix=${PREFIX} install
  # Gets installed by binutils:
  rm ${PREFIX}/share/info/bfd.info
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

