set -e -x

CHOST=$(${SRC_DIR}/build/gcc-final/gcc/xgcc -dumpmachine)
export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/buildtools/bin:${SRC_DIR}/compilers/bin:${PATH}

pushd ${SRC_DIR}/build/duma-target
  make prefix=${PREFIX} HOSTCC=$(uname -m)-build_pc-linux-gnu-gcc CC=${CHOST}-gcc CXX=${CHOST}-g++ RANLIB=${CHOST}-ranlib OS=linux DUMA_CPP=1 install
popd

source ${RECIPE_DIR}/install_scripts/make_tool_links.sh
