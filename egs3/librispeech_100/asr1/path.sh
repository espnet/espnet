#!/bin/bash

export HF_HOME=~/workspace/espnet3/hub
export PYTHONPATH=${PYTHONPATH}:../../../
. ../../../tools/activate_python.sh

export LIBRISPEECH=data/librispeech

export N_GPU=1
