#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(
  cat <<EOF
Usage: $0

Options:
    --include_control (bool): Whether to include control group data
    --include_aphasia_type (bool): Whether to include aphasia type in the beginning of each sentence ("[wernicke]")
EOF
)
SECONDS=0

stage=1
stop_stage=100
include_control=false
include_aphasia_type=false

log "$0 $*"
. utils/parse_options.sh

if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${APHASIABANK}" ]; then
  log "Error: \$APHASIABANK is not set in db.sh."
  exit 2
fi

tmp=data/local
mkdir -p $tmp

# Things to manually prepare:
# - Download AphasiaBank data from https://aphasia.talkbank.org/
# - Set ${APHASIABANK} to the path to English subset ("<data_root>/English/") in db.sh
# - Download transcripts (Aphasia and Control subfolder) from
#   https://aphasia.talkbank.org/data/English/
# - Unzip and copy all *.cha files into ${APHASIABANK}/transcripts

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Converting *.mp4 and *.mp3 files into .wav"

  for ext in mp3 mp4; do
    for subdir in Aphasia Control; do
      for f in $(find ${APHASIABANK}/${subdir} -type f -name "*.${ext}"); do
        filename=$(basename -- "$f")
        dir=$(dirname "$f")
        filename="${filename%.*}"
        echo "Converting $f to $dir/${filename}.wav"
        ffmpeg -y -i "$f" -acodec pcm_s16le -ac 1 -ar 16000 "${dir}/${filename}.wav" &>/dev/null
      done
    done
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Extracting speaker information"

  # generate data/spk_info.txt

  # install pylangacq
  pip install --upgrade pylangacq

  python local/extract_speaker_info.py --transcript-dir=${APHASIABANK}/transcripts --out-dir=data
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log "Extracting sentence information"

  # generate data/local/text
  _opts="--transcript-dir=${APHASIABANK}/transcripts --out-dir=$tmp "
  if "${include_aphasia_type}"; then
    _opts+="--spk2aphasia-type=data/spk2aphasia_type "
  fi

  python local/extract_sentence_info.py ${_opts}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log "Split data into train, test, and val"

  # split data, generate text and utt2spk
  python local/split_train_test_val.py --text=$tmp/text --out-dir=$tmp
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  log "Generating data files of the entire database"

  # generate wav.scp of all data
  find ${APHASIABANK}/Aphasia -type f -name "*.wav" >$tmp/all_wav.list

  # add control group data if needed
  if "${include_control}"; then
    log "**Including the control group**"
    find ${APHASIABANK}/Control -type f -name "*.wav" >>$tmp/all_wav.list
  fi

  awk -F'/' '{printf("%s\t%s\n",$NF,$0)}' $tmp/all_wav.list >$tmp/all_wav.scp
  sed -i 's/\.wav//' $tmp/all_wav.scp
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  log "Generating data files of subsets"

  # generate 'wav.scp' and 'segments' files for subsets
  for x in train val test; do
    python local/generate_wavscp_and_segments.py \
      --wavscp=$tmp/all_wav.scp \
      --reco-list=$tmp/$x/utt.list \
      --out-dir="$tmp/$x/"
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  log "Finalizing data"

  # finalize
  for x in train val test; do
    cp -r $tmp/$x data/
    utils/fix_data_dir.sh data/$x
  done
fi
