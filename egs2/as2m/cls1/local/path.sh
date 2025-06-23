export NCCL_SOCKET_IFNAME=hsn
module load nccl
module load gcc/11.4.0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
