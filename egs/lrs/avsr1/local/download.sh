#! /usr/bin/env bash 

# Copyright 2020 Ruhr-University (Wentao Yu)

. ./cmd.sh
. ./path.sh

git clone https://github.com/rub-ksv/lrs_avsr1_local.git
for file in data_prepare dump extract_reliability training; do
	cp -R lrs_avsr1_local/$file local
done
rm -rf lrs_avsr1_local
exit 0
