# If project is in ESPnet directory (e.g. espnet/egs/project/avsr1), the MAIN_ROOT path can be set with:
MAIN_ROOT=$PWD/../../.. 
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
ESPNET_VENV="espnet_venv"

# Setting path variables for dataset, OpenFace, DeepXi, pretrained model and musan
# Change this variables and adapt it to your Folder structure
export DATA_DIR="/home/foo/LRS2"					# The LRS2 dataset directory
export DATALRS3_DIR="/home/foo/LRS3"					# The LRS3 dataset directory
export OPENFACE_DIR="/home/foo/AVSR/OpenFace/build/bin"			# OpenFace build directory
export VIDAUG_DIR="/home/foo/ESPnet/install/vidaug" 			# Path to vidaug directory if it is not installed in espnet virtual environment
export DEEPXI_DIR="/home/foo/AVSR/DeepXi" 				# DeepXi directory
export DEEPXI_VENVDIR="/home/foo/venv/DeepXi/bin/activate"		# DeepXi virtual environment directory 
export PRETRAINEDMODEL="pretrainedvideomodel/Video_only_model.pt"	# Path to pretrained video model
export MUSAN_DIR="musan"   						#  The noise dataset directory 

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh

export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH
export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

if [ ! -L ./steps ]; then
    ln -s  $KALDI_ROOT/egs/wsj/s5/steps ./steps
fi
if [ ! -L ./utils ]; then
    ln -s  $KALDI_ROOT/egs/wsj/s5/utils ./utils
fi

