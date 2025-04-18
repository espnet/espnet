#!/bin/bash

export HF_HOME=~/workspace/espnet3/hub
. ../../../tools/activate_python.sh

export LIBRISPEECH_PATH=download/LibriSpeech
export LIBRISPEECH_100=download/librispeech_dataset
export LIBRISPEECH=download/librispeech_dataset
export LIBRISPEECH_960=download/librispeech_dataset_960

export N_GPU=1
