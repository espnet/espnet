#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--sample_rate <16k/48k>]
EOF
)


sample_rate=16k
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ! -e "${CHIME1}" ]; then
    log "Fill the value of 'CHIME1' in db.sh"
    log "You can download the data manually from https://spandh.dcs.shef.ac.uk/chime_challenge/chime2011/datasets.html"
    exit 1
fi
if [ ${sample_rate} != "16k" ] && [ ${sample_rate} != "48k" ]; then
    log "Invalid sample rate: ${sample_rate} (must be 16k or 48k)"
    exit 1
fi

train_set=train_${sample_rate}
valid_set=devel_${sample_rate}
test_set=test_${sample_rate}

# Setup wav folders
wav_train="${CHIME1}/PCCdata${sample_rate}Hz/train/reverberated"
wav_devel="${CHIME1}/PCCdata${sample_rate}Hz/devel/isolated"
wav_test="${CHIME1}/PCCdata${sample_rate}Hz/test/isolated"
noise_train="${CHIME1}/PCCdata${sample_rate}Hz/train/background"
noise_devel="${CHIME1}/PCCextradata${sample_rate}Hz/backgrounds/devel"
noise_test="${CHIME1}/PCCdata${sample_rate}Hz/test/embedded"
annotation_test="${CHIME1}/test"

if [ ! -d "$wav_train" ]; then
  echo "Cannot find wav directory $wav_train"
  echo "Please ensure the tar balls in '$CHIME1' have been extracted"
  exit 1;
fi
set_list="${train_set}"
if [ -d "$wav_devel" ]; then
  set_list="${set_list} ${valid_set}"
  mkdir -p "data/${valid_set}"

  if [ ! -d "$noise_devel" ]; then
    log "Cannot find the noise directory for devel: $noise_devel"
    exit 2
  fi
fi
if [ -d "$wav_test" ]; then
  set_list="${set_list} ${test_set}"
  mkdir -p "data/${test_set}"

  if [ ! -d "$noise_test" ]; then
    log "Cannot find the noise directory for test: $noise_test"
    exit 2
  fi
  if [ ! -d "$annotation_test" ]; then
    log "Cannot find the annotation directory for test: $annotation_test"
    exit 2
  fi
fi
echo "Preparing data sets: $set_list"


log "Data preparation for training data"
mkdir -p "data/${train_set}"
# Create spk1.scp
scp="data/${train_set}/spk1.scp"
rm -f "$scp" || true
for sid in $(seq 34); do
  sid2=$(printf "s%02d" $sid)
  # shellcheck disable=SC2012
  ls -1 "${wav_train}/id${sid}"/*.wav \
    | perl -ape "s/(.*)\/(.*).wav/${sid2}_\2\t\1\/\2.wav/;" \
    | sort >> $scp
done
rm -f "data/${train_set}/wav.scp" || true
ln -s spk1.scp "data/${train_set}/wav.scp"
# Create noise_list.scp
# shellcheck disable=SC2012
ls -1 "${noise_train}"/*.wav | sort > "data/${train_set}/noise_list.scp"


log "Data preparation for devel and test data"
for x in "${valid_set}" "${test_set}"; do
  if [ -d "data/${x}" ]; then
    # Create wav.scp
    scp="data/${x}/wav.scp"
    rm -f "$scp" || true
    if [ "$x" = "${valid_set}" ]; then
      wav_dir="${wav_devel}"
    else
      wav_dir="${wav_test}"
    fi
    for sid in $(seq 34); do
      sid2=$(printf "s%02d" $sid)
      # shellcheck disable=SC2012
      ls -1 "${wav_dir}"/{0dB,3dB,6dB,9dB,m3dB,m6dB}/s${sid}_*.wav \
        | perl -ape "s/(.*)\/(.*)\/s.*_(.*).wav/${sid2}_\3_\2\t\1\/\2\/s${sid}_\3.wav/;" \
        | sort >> $scp
    done

    if [ "$x" = "${valid_set}" ]; then
      sed -e "s#/[0-9]\+dB/#/clean/#g" "data/${x}/wav.scp" > "data/${x}/spk1.scp"

      # Create noise1.scp
      python local/prepare_devel_noise1_scp.py "$scp" "$noise_devel" --outfile "data/${x}/noise1.scp"
    else
      # spk1.scp is not available for test data

      # Create noise1.scp
      python local/prepare_test_noise1_scp.py "$scp" "$noise_test" "$annotation_test" --outfile "data/${x}/noise1.scp"
    fi
  fi
done


log "Preparing other files (text, utt2spk, spk2utt)"
# Prepare other files in data/setname/
for x in $set_list; do
  scp="data/${x}/wav.scp"
  if [ -f "$scp" ]; then
    # Create transcription files
    cut -f1 $scp | local/create_chime1_trans.pl - > "data/${x}/text"

    # Create utt2spk files
    # No speaker ID
    # perl -ape "s/(.*)\t.*/\1\t\1/;" < "$scp" > "data/${x}/utt2spk"
    # Use speaker ID
    perl -ape "s/(s..)(.*)\\t.*/\1\2\t\1/;" < "$scp" > "data/${x}/utt2spk"

    # Create spk2utt files
    utils/utt2spk_to_spk2utt.pl "data/${x}/utt2spk" > "data/${x}/spk2utt" || exit 1;
  fi
done


log "Successfully finished. [elapsed=${SECONDS}s]"
