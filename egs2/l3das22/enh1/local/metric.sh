#!/usr/bin/env bash

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


echo "Apply jiwer and transformers for calculating the official metric of the L3DAS22 challenge Task1"
python -m pip show -f jiwer >/dev/null || python -m pip install jiwer
python -m pip show -f transformers >/dev/null || python -m pip install transformers

if [ $# -ne 2 ]; then
    echo "Usage: local/metric.sh <predicted scp> <target scp>"
    exit 1;
fi

if [ $# -eq 2 ]; then
    predicted_path=$1
    target_path=$2
    python local/metric.py --predicted_path ${predicted_path} --target_path ${target_path}
else
    python local/metric.py
fi

exit 0
