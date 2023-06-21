#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

zipped_data=$1
raw_data=$2/Datatang-English/data

# unzip and rename each accent
unzip $zipped_data -d ${2}
mv $raw_data/American\ English\ Speech\ Data $raw_data/US
mv $raw_data/British\ English\ Speech\ Data $raw_data/UK
mv $raw_data/Chinese\ Speaking\ English\ Speech\ Data $raw_data/CHN 
mv $raw_data/Indian\ English\ Speech\ Data $raw_data/IND 
mv $raw_data/Portuguese\ Speaking\ English\ Speech\ Data $raw_data/PT 
mv $raw_data/Russian\ Speaking\ English\ Speech\ Data $raw_data/RU 
mv $raw_data/Japanese\ Speaking\ English\ Speech\ Data $raw_data/JPN 
mv $raw_data/Korean\ Speaking\ English\ Speech\ Data $raw_data/KR
mv $raw_data/Canadian\ English\ Speech\ Data $raw_data/CA 
mv $raw_data/Spanish\ Speaking\ English\ Speech\ Data $raw_data/ES

echo "local/download_and_untar.sh succeeded"
exit 0;
