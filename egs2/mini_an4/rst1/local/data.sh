#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100
an4_root=./downloads/an4

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        tar -xvf downloads.tar.gz
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train,test}
    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi
    python3 local/data_prep.py ${an4_root} sph2pipe
    for x in train test; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done

    # mini_an4 is clean speech: it is the clean target of the feature
    # predictor (degraded online in training) and of the vocoder. The six
    # training utterances also serve as the validation set.
    for x in train_fp dev_fp; do
        utils/copy_data_dir.sh data/train data/${x}
    done
    # The vocoder synthesises 48 kHz audio, so its sets are resampled to 48 kHz.
    for x in train_voc dev_voc; do
        scripts/audio/format_wav_scp.sh --nj 1 --cmd "${train_cmd}" \
            --fs 48000 --audio-format wav data/train/wav.scp data/${x}
    done

    # Noise pool for online degradation, from the noise clips in downloads/.
    mkdir -p data/noise_pool
    cp downloads/noise/*.wav data/noise_pool/
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Tiny w2v-BERT 2.0 backbone"
    # The real backbone (facebook/w2v-bert-2.0) is far too large for CI and no
    # official tiny checkpoint exists, so a randomly initialised one with the
    # same architecture is written locally; conf/*_debug.yaml load it.
    python3 local/make_tiny_w2v_bert.py --out_dir data/tiny_w2v_bert
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
