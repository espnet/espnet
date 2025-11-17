#!/bin/bash

source path.sh

python run.py \
    --stage all \
    --train_config conf/tuning/train_e_branchformer.yaml \
    --eval_config conf/evaluate.yaml \
    \
    --stage.create_dataset.func src.create_dataset.create_dataset \
    --stage.train_tokenizer.dataset_dir /path/to/LibriSpeech/ \
    --stage.train.dataset_dir /path/to/LibriSpeech/ \
    # --dry_run
