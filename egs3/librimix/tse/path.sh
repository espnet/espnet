#!/bin/bash


export PYTHONPATH=../../../:../../TEMPLATE/tse:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh

# Set this to the directory path containing the unpacked LibriSpeech corpus.
#     You will need to download it manually
export LIBRISPEECH=
# Set this to the directory containing the LibriMix dataset.
#     If not set, the script will default to 'data/LibriMix'.
export LIBRIMIX=

# Keep the CPU math libraries single-threaded. PyTorch and OpenBLAS size their
# thread pools from the host core count, which ignores a cgroup CPU quota, so in
# a container they oversubscribe badly (e.g. 64 threads against a 16-CPU quota).
# OpenMP threads spin rather than sleep after a parallel region, so the idle pool
# burns the quota and CFS then freezes the whole cgroup - DataLoader workers
# included - until the next period. That shows up as a large `iter_time`.
# These are inherited by the forked DataLoader workers, which is what we want.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
