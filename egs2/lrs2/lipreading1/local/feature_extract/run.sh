#!/bin/bash

split_id=$1

`which python` ./feature_extract.py \
    --path ../lrw_pretrain/fine4/finetuneGRU_every_frame/finetuneGRU_19.pt \
    --batch-size 1\
    --source-dir $split_id \
    --target-dir ../dataset/LRS2_vfeature \
    --crop-size 102 \
    --workers 16

