#!/usr/bin/env bash
MAIN_ROOT=$PWD/../../..
KALDI_ROOT=${MAIN_ROOT}/tools/kaldi

[ -f ${KALDI_ROOT}/tools/env.sh ] && . ${KALDI_ROOT}/tools/env.sh
export PATH=$PWD/utils/:${KALDI_ROOT}/tools/openfst/bin:${KALDI_ROOT}/tools/sctk/bin:$PWD:$PATH
[ ! -f ${KALDI_ROOT}/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. ${KALDI_ROOT}/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MAIN_ROOT}/tools/chainer_ctc/ext/warp-ctc/build
source ${MAIN_ROOT}/tools/venv/bin/activate
export PATH=${MAIN_ROOT}/utils:${MAIN_ROOT}/espnet/bin:$PATH

export OMP_NUM_THREADS=1
