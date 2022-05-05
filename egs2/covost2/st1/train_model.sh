#!/usr/bin/env bash
module load cuda/10.2.0 cudnn mkl
cd /ocean/projects/cis210027p/siddhana/new_download/espnet/egs2/covost2/st1
./run.sh >> result.txt
