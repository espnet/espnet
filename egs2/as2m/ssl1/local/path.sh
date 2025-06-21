#export NCCL_P2P_DISABLE=1

#export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
# export NCCL_DEBUG=INFO

# THESE WORK WITH SINGLE NODE
export NCCL_SOCKET_IFNAME=hsn
module load nccl
module load gcc/11.4.0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# THESE are added for testing MULTI NODE
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_ASYNC_ERROR_HANDLING=1

# export CUDA_LAUNCH_BLOCKING=1