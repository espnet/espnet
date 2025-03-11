#!/usr/bin/env bash

# Copyright 2024 Carnegie Mellon University (William Chen)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

mkdir -p data/train
mkdir -p data/dev
mkdir -p data/dev_dialect
mkdir -p data/raw_audio
mkdir -p data/local

python local/download.py
utils/fix_data_dir.sh data/train
utils/fix_data_dir.sh data/dev
utils/fix_data_dir.sh data/dev_dialect
