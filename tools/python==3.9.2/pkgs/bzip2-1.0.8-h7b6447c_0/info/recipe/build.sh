#!/bin/sh

USED_CC=${GCC:-${CC}}
make install PREFIX=${PREFIX} CFLAGS="${CFLAGS}" CC="${USED_CC}"

# build shared library
if [[ ${target_platform} =~ .*linux*. ]]; then
  make -f Makefile-libbz2_so CFLAGS="${CFLAGS}" CC="${USED_CC}"
  ln -s libbz2.so.${PKG_VERSION} libbz2.so
  cp -d libbz2.so* ${PREFIX}/lib/
elif [[ ${target_platform} == osx-64 ]]; then
  ${USED_CC} -shared -Wl,-install_name -Wl,libbz2.dylib -o libbz2.${PKG_VERSION}.dylib blocksort.o huffman.o crctable.o randtable.o compress.o decompress.o bzlib.o
  cp libbz2.${PKG_VERSION}.dylib ${PREFIX}/lib/
  ln -s libbz2.${PKG_VERSION}.dylib ${PREFIX}/lib/libbz2.dylib
fi
