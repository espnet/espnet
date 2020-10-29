MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# check extra module installation
if ! command -v parallel-wavegan-train > /dev/null; then
    echo "Error: parallel_wavegan is not installed." >&2
    echo "Error: Please install via \`. ./path.sh && pip install -U parallel_wavegan\`" >&2
    return 1
fi
if ! python3 -c "import pytorch_lamb" > /dev/null; then
    echo "Error: pytorch_lamb is not installed." >&2
    echo "Error: Please install via https://github.com/cybertronai/pytorch-lamb" >&2
    return 1
fi

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
