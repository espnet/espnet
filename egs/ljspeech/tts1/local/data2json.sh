#!/bin/bash

set -e

# ESPnet dep: merge_scp2json.py, feat-to-shape.py, text2shape.py

data_dir=$1
dump_dir=$2
frontend=$3
data_json=$4

# feats.scp and feats_shape.scp
feats=$dump_dir/feats.scp
feats_shape=$dump_dir/feats_shape.scp
feat-to-shape.py scp:$feats > $feats_shape

# text.scp and text_shape.scp
text=$data_dir/text
text_shape=$dump_dir/text_shape.scp
text2shape.py $text --frontend $frontend > $text_shape

# make data.json
merge_scp2json.py \
  --input-scps feat:$feats shape:$feats_shape:shape \
  --output-scps text:$text shape:$text_shape:shape \
  > $data_json

rm -f $feats_shape $text_shape
