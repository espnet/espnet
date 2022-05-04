#!/usr/bin/env bash

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/metric.sh <predicted scp> <target scp>"
  exit 1;
fi

if [ $# -gt 1 ]; then
	predicted_path=$1
	target_path=$2
  python local/metrics.py --predicted_path ${predicted_path} --target_path ${target_path}
else
	python local/metrics.py
fi

exit 0

