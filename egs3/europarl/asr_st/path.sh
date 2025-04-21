#!/bin/bash

# make sure we can use ffmpeg to load m4a audio
export PATH=${PATH}:../../../tools/ffmpeg-release

# number of GPU, if we want to control from environment variable.
export N_GPU=1
