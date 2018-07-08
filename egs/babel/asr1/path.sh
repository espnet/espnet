MAIN_ROOT=/mnt/matylda3/karafiat/BABEL/GIT/espnet.github.v2
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
SPNET_ROOT=$MAIN_ROOT/src

CONDADIR=/homes/kazi/karafiat/local/anaconda2
CUDA_ROOT=/usr/local/share/cuda-9.0.176

export CUDA_HOME=$CUDA_ROOT
export CUDA_PATH=$CUDA_ROOT

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$CUDA_ROOT/bin:$PWD:$PATH
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=$SPNET_ROOT/utils/:$SPNET_ROOT/bin/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build:$CUDA_ROOT/lib64
source $CONDADIR/bin/activate py27_cuda9
export PYTHONPATH=$CONDADIR/envs/py27_cuda9/bin:$SPNET_ROOT/lm/:$SPNET_ROOT/asr/:$SPNET_ROOT/nets/:$SPNET_ROOT/utils/:$SPNET_ROOT/bin/:$PYTHONPATH

export OMP_NUM_THREADS=1
export CC=/usr/local/bin/gcc-5.3
export CXX=/usr/local/bin/g++-5.3

export CPATH=$CUDA_ROOT/include
