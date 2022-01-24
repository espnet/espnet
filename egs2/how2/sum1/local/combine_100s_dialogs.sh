#! /bin/bash


dset=

. utils/parse_options.sh || exit 1



mkdir -p data/${dset}_vid

cp data/${dset}/wav.scp data/${dset}_vid/
