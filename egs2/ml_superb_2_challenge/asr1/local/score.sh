#!/usr/bin/env bash

# Copyright 2024 Carnegie Mellon University (William Chen)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
asr_exp=$1

python local/score.py --exp_dir $asr_exp
