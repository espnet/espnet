MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C



. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh

export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export LD_LIBRARY_PATH=$MAIN_ROOT/tools/lib:$MAIN_ROOT/tools/lib64:$LD_LIBRARY_PATH

# check extra module installation
if ! python3 -c "import pyopenjtalk" > /dev/null; then
    echo "Error: pyopenjtalk is not installed." >&2
    echo "Error: please install pyopenjtalk and its dependencies as follows:" >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make pyopenjtalk.done" >&2
    return 1
fi

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8


# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
