#!/usr/bin/env bash
set -euo pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "$0" "$@"

downloads_dir=

. ./utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
  log "Error: Unknown argument $*"
  cat <<EOF
  Usage: local/data.sh [--downloads_dir <path>]

  Options:
    --downloads_dir : Directory that contains TAL_ASR-*.zip files.
EOF
  exit 1
fi

if [ -z "${TAL_ZH_ADULT_TEACH}" ]; then
  log "Error: \$TAL_ZH_ADULT_TEACH is not set in db.sh."
  exit 2
fi

log "Download data to ${TAL_ZH_ADULT_TEACH}"
TAL_ZH_ADULT_TEACH=$(cd "${TAL_ZH_ADULT_TEACH}"; pwd)
. ./local/download_and_untar.sh "${TAL_ZH_ADULT_TEACH}" "${downloads_dir}"

prepare_kaldi() {
  name=$1
  wav_dir=$TAL_ZH_ADULT_TEACH/$2
  trans=$TAL_ZH_ADULT_TEACH/$3
  num_wav=$4
  dir=data/${name}
  log "Data Preparation for partition: data/${name}"
  mkdir -p "$dir"

  find "${wav_dir}" -name "*.wav" > "$dir"/wav.flist

  sed -e 's/\.wav//' "$dir"/wav.flist | awk -F '/' '{print $NF}' > "$dir"/utt.list
  sed -e 's/\.wav//' "$dir"/wav.flist | awk -F '/' '{print $NF, "TALASR"$(NF-1)"-"$NF}' > "$dir"/utt_uttid
  sed -e 's/\.wav//' "$dir"/wav.flist | awk -F '/' '{print "TALASR"$(NF-1)"-"$NF, "TALASR"$(NF-1)}' > "$dir"/utt2spk
  paste -d ' ' <(awk '{print $2}' "$dir"/utt_uttid) "$dir"/wav.flist > "$dir"/wav.scp
  utils/filter_scp.pl -f 1 "$dir"/utt.list "$trans" | \
    sed 's/Ａ/A/g' | sed 's/#//g' | sed 's/=//g' | sed 's/、//g' | \
    sed 's/，//g' | sed 's/？//g' | sed 's/。//g' | sed 's/[ ][ ]*$//g'\
    > "$dir"/transcripts.txt
  awk '{print $1}' "$dir"/transcripts.txt > "$dir"/utt.list
  paste -d " " <(sort -u -k 1 "$dir"/utt_uttid | awk '{print $2}') \
    <(sort -u -k 1 "$dir"/transcripts.txt | awk '{for(i=2;i<NF;i++) {printf($i" ")}printf($NF"\n") }') \
    > "$dir"/text
  utils/utt2spk_to_spk2utt.pl "$dir"/utt2spk > "$dir"/spk2utt

  if [ "$(wc -l < "$dir/text")" -ne "$num_wav" ]; then
    log "Error: The number of utterances in $dir/text ($(wc -l < "$dir/text")) does not match the expected number ($num_wav)."
    exit 1
  fi

  utils/fix_data_dir.sh "$dir"

}

while read -r name wav_subdir trans num_wav; do
  prepare_kaldi "$name" "$wav_subdir" "$trans" "$num_wav"
done <<EOF
train_1 aisolution_data/wav/train aisolution_data/transcript/transcript.txt 22467
dev     aisolution_data/wav/dev   aisolution_data/transcript/transcript.txt 3208
test    aisolution_data/wav/test  aisolution_data/transcript/transcript.txt 6072
train_2 CH                    CH/CH_transcript.txt                     29386
train_3 MA                    MA/MA_transcript.txt                     38924
EOF

utils/combine_data.sh data/train data/train_1 data/train_2 data/train_3

utils/validate_data_dir.sh --no-feats data/train
utils/validate_data_dir.sh --no-feats data/dev
utils/validate_data_dir.sh --no-feats data/test

log "Successfully finished. [elapsed=${SECONDS}s]"
exit 0;
