#!/bin/bash

. ./path.sh

WSJ_REVERB="/home/chenda/workspace/cmu/sgmse/wsj_reverb/audio"


for f in test valid train;
do

mkdir -p data/${f}
find ${WSJ_REVERB}/${f}/noisy/*.wav | sort > data/${f}/wav.id.scp
find ${WSJ_REVERB}/${f}/clean/*.wav | sort > data/${f}/spk1.id.scp

paste <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/wav.id.scp) > data/${f}/wav.scp
paste <(cat data/${f}/spk1.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/spk1.id.scp) > data/${f}/spk1.scp

paste <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) > data/${f}/utt2spk
cp data/${f}/utt2spk data/${f}/spk2utt

done