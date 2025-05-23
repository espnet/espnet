#!/bin/bash

export HF_HOME=~/workspace/espnet3/hub
export PYTHONPATH=${PYTHONPATH}:../../../
source ../../../tools/activate_python.sh

export LIBRISPEECH=data/librispeech

export WANDB_API_KEY=0cdbe23a746157c701dfa8ea691597ad65e2d400
wandb login
