#!/usr/bin/env bash

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

pip show -f jiwer >/dev/null || pip install jiwer
pip show -f pystoi >/dev/null || pip install pystoi
pip show -f transformers >/dev/null || pip install transformers

if [ $# -lt 1 ]; then
    echo "Usage: local/metric.sh <predicted scp> <target scp>"
    exit 1;
fi

if [ $# -gt 1 ]; then
    predicted_path=$1
    target_path=$2
    python local/metric.py --predicted_path ${predicted_path} --target_path ${target_path}
else
    python local/metric.py
fi

exit 0

