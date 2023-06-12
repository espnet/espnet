#!/bin/bash

set -ex

# debugging ...
if [ -z "${RECIPE_DIR}" ]; then
    RECIPE_DIR=$PWD
fi

if [[ ${target_platform} =~ osx-.* ]]; then
  if [[ ! -f ${BUILD_PREFIX}/bin/llvm-objcopy ]]; then
    echo "no llvm-objcopy"
    exit 1
  fi
  ln -s ${BUILD_PREFIX}/bin/llvm-objcopy ${BUILD_PREFIX}/bin/x86_64-apple-darwin19.6.0-objcopy
  chmod +x ${BUILD_PREFIX}/bin/x86_64-apple-darwin19.6.0-objcopy
  ln -s ${BUILD_PREFIX}/bin/llvm-objcopy ${BUILD_PREFIX}/bin/objcopy
  chmod +x ${BUILD_PREFIX}/bin/objcopy
  unset CC CXX
fi

if [[ $(uname) == Linux ]]; then
  ulimit -s 32768 || true
fi

if [[ "${bootstrapping}" != "yes" ]]; then
  # Use own compilers instead of relying on ones from the docker image/system
  conda create -p $SRC_DIR/compilers gcc_${target_platform} gxx_${target_platform} binutils \
                                     make gawk sed libtool --yes --quiet
fi

${RECIPE_DIR}/build_scripts/build.sh

exit 0
