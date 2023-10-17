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
random_enrollment=false

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

train_set="train_nodev"
train_dev="train_dev"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        local/asr_data.sh --stage 1 --stop-stage 1 --an4-root "${an4_root}"
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    if [ ! -d data/train ] || [ ! -d data/test ]; then
        local/asr_data.sh --stage 2 --stop-stage 2 --an4-root "${an4_root}"
    fi

    rm data/**/utt2category 2>/dev/null || true
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Enrollment data preparation"
    for dset in ${train_set} ${train_dev} test; do
        scripts/audio/format_wav_scp.sh --nj 4 --cmd "${train_cmd}" \
            --out-filename "wav.scp" \
            --audio-format "flac" --fs "16k" \
            "data/${dset}/wav.scp" "dump/raw/org/${dset}" \
            "dump/raw/org/${dset}/logs/wav" "dump/raw/org/${dset}/data/wav"
     done

    python3 local/prepare_spk2enroll_mini_an4.py \
        "dump/raw/org/${train_set}" \
        --outfile data/${train_set}/spk2enroll.json \
        --audio_format flac

    python3 local/prepare_spk2enroll_mini_an4.py \
        "dump/raw/org/${train_set}" "dump/raw/org/${train_dev}" \
        --outfile data/${train_dev}/spk2enroll.json \
        --audio_format flac

    python3 local/prepare_spk2enroll_mini_an4.py \
        "dump/raw/org/test" \
        --outfile data/test/spk2enroll.json \
        --audio_format flac

    for dset in ${train_set} ${train_dev} test; do
        if [ "${dset}" = "${train_set}" ] && $random_enrollment; then
            is_train=True
        else
            is_train=False
        fi
        # This script generates enroll_spk?.scp under "data/${dset}"
        python local/prepare_mini_an4_enroll.py \
            data/${dset}/wav.scp \
            data/${dset}/spk2enroll.json \
            --train ${is_train} \
            --seed 1 \
            --output_dir data/${dset} \
            --outfile_prefix "enroll_spk"
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Data preparation for variable numbers of speakers (up to 3 speakers)"
    for dset in ${train_set} ${train_dev} test; do
        mkdir -p "data/${dset}_unk_nspk"
        cp "data/${dset}/wav.scp" "data/${dset}_unk_nspk"
        cp "data/${dset}/utt2spk" "data/${dset}_unk_nspk"
        cp "data/${dset}/spk2utt" "data/${dset}_unk_nspk"
        awk -v min=1 -v max=3 '{srand(FNR*6); nspk=int(min+rand()*(max-min+1)); print($1 " " nspk "spk")}' \
            "data/${dset}/enroll_spk1.scp" > "data/${dset}_unk_nspk/utt2category"
        # NOTE: Kaldi pipe IO is not supported when preparing the multi-column scp file
        awk 'NR==FNR{nspk[$1]=substr($2,1,1); next} {msg=$1; for (i=1; i<=nspk[$1]; ++i) {msg=msg" "$2} print(msg)}' \
            "data/${dset}_unk_nspk/utt2category" "dump/raw/org/${dset}/wav.scp" > "data/${dset}_unk_nspk/spk1.scp"
        if [ "${dset}" = "${train_set}" ] && $random_enrollment; then
            awk 'NR==FNR{nspk[$1]=substr($2,1,1); next} {msg=$1; s=$2; for (i=3; i<=NF; i++) {s=s" "$i;} for (i=1; i<=nspk[$1]; ++i) {msg=msg" "s} print(msg)}' \
            "data/${dset}_unk_nspk/utt2category" "data/${dset}/enroll_spk1.scp" > "data/${dset}_unk_nspk/enroll_spk1.scp"
        else
            cp "data/${dset}_unk_nspk/spk1.scp" "data/${dset}_unk_nspk/enroll_spk1.scp"
        fi
        cp "data/${dset}_unk_nspk/spk1.scp" "data/${dset}_unk_nspk/dereverb1.scp"
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
