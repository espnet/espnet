# check extra module installation
if ! python3 -c "import fairseq" > /dev/null; then
    echo "Error: fairseq is not installed." >&2
    echo "Error: please install fairseq and its dependencies as follows:" >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make fairseq.done" >&2
    return 1
fi

#export NCCL_P2P_DISABLE=1

#export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
module load nccl
module load gcc/11.4.0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

