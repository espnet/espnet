set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
declare -a COMMON_MAKE_OPTS=()
COMMON_MAKE_OPTS+=(prefix=${PREFIX} exec_prefix=${PREFIX})

_libdir=libexec/gcc/${CHOST}/${PKG_VERSION}
# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/gcc-final/

# adapted from Arch install script from https://github.com/archlinuxarm/PKGBUILDs/blob/master/core/gcc/PKGBUILD
# We cannot make install since .la files are not relocatable so libtool deliberately prevents it:
# libtool: install: error: cannot install `libgfortran.la' to a directory not ending in ${SRC_DIR}/work/gcc_built/${CHOST}/lib/../lib
make -C ${CHOST}/libgfortran "${COMMON_MAKE_OPTS[@]}" all-multi libgfortran.spec ieee_arithmetic.mod ieee_exceptions.mod ieee_features.mod config.h
make -C gcc "${COMMON_MAKE_OPTS[@]}" fortran.install-{common,man,info}

# How it used to be:
# install -Dm755 gcc/f951 ${PREFIX}/${_libdir}/f951
for file in f951; do
  if [[ -f gcc/${file} ]]; then
    install -c gcc/${file} ${PREFIX}/${_libdir}/${file}
  fi
done

cp ${CHOST}/libgfortran/libgfortran.spec ${PREFIX}/${CHOST}/sysroot/lib64

pushd ${PREFIX}/bin
  ln -s ${CHOST}-gfortran ${CHOST}-f95
popd

popd

mkdir -p $PREFIX/lib/gcc/${CHOST}/${gcc_version}/finclude
rsync -av ${SRC_DIR}/gcc_built/lib/gcc/${CHOST}/${gcc_version}/finclude/ $PREFIX/lib/gcc/${CHOST}/${gcc_version}/finclude

# Install Runtime Library Exception
install -Dm644 $SRC_DIR/gcc/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/gcc-fortran/RUNTIME.LIBRARY.EXCEPTION

# generate specfile so that we can patch loader link path
# link_libgcc should have the gcc's own libraries by default (-R)
# so that LD_LIBRARY_PATH isn't required for basic libraries.
#
# GF method here to create specs file and edit it.  The other methods
# tried had no effect on the result.  including:
#   setting LINK_LIBGCC_SPECS on configure
#   setting LINK_LIBGCC_SPECS on make
#   setting LINK_LIBGCC_SPECS in gcc/Makefile
specdir=`dirname $($PREFIX/bin/${CHOST}-gfortran -print-libgcc-file-name -no-canonical-prefixes)`
mv $PREFIX/bin/${CHOST}-gfortran $PREFIX/bin/${CHOST}-gfortran.bin
echo '#!/bin/sh' > $PREFIX/bin/${CHOST}-gfortran
echo $PREFIX/bin/${CHOST}-gfortran.bin -specs=$specdir/specs '"$@"' >> $PREFIX/bin/${CHOST}-gfortran
chmod +x $PREFIX/bin/${CHOST}-gfortran

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
