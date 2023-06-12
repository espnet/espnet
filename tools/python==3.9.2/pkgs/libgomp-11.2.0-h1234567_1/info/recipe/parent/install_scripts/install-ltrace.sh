set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/ltrace-target
  make DESTDIR="${PREFIX}" install
popd

source ${RECIPE_DIR}/install_scripts/make_tool_links.sh

