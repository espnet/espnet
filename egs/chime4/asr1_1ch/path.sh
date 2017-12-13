MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
JQ_ROOT=$MAIN_ROOT/tools/jq
SPNET_ROOT=$MAIN_ROOT/src

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=$SPNET_ROOT/utils/:$SPNET_ROOT/bin/:$JQ_ROOT:$PATH
source $MAIN_ROOT/tools/venv/bin/activate
export PYTHONPATH=$SPNET_ROOT/nets/:$SPNET_ROOT/utils/:$SPNET_ROOT/bin/:$PYTHONPATH
