MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $MAIN_ROOT/tools/venv/bin/activate
fi
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH
export PATH=$MAIN_ROOT/tools/PESQ/P862/Software/source:$PATH

export OMP_NUM_THREADS=1

# check extra module installation
if ! which sox > /dev/null; then
    echo "Error: it seems that sox is not installed." >&2
    echo "Error: please install sox." >&2
    if conda &> /dev/null; then
        echo "Error: you can install sox using conda." >&2
        echo "Error: conda install -c conda-forge sox" >&2
    fi
    return 1
fi
if ! which PESQ > /dev/null; then
    echo "Error: it seems that PESQ is not installed." >&2
    echo "Error: please install PESQ as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make pesq" >&2
    return 1
fi
