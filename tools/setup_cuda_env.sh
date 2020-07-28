
if [ $# -eq 0 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 CUDA_HOME [NCCL_HOME]"
    return 1
fi
export CUDA_HOME="$1"
echo "CUDA_HOME=${CUDA_HOME}"

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export CPATH=$CUDA_HOME/include:$CPATH
export CUDA_PATH=$CUDA_HOME

if [ $# -eq 2 ]; then
    NCCL_HOME="$2"
    echo "NCCL_HOME=${NCCL_HOME}"
    export CPATH=$NCCL_HOME/include:$CPATH
    export LD_LIBRARY_PATH=$NCCL_HOME/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$NCCL_HOME/lib/:$LIBRARY_PATH
fi
