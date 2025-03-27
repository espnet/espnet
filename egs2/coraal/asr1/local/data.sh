#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# adapted from egs2/aishell/asr1/local/data.sh
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

# Data preparation related
data_url=http://lingtools.uoregon.edu/coraal/coraal_download_list.txt
remove_archive=false
download_opt=


log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if "$remove_archive"; then
  download_opt="--remove-archive"
fi

if [ -z "${CORAAL}" ]; then
  log "Error: \$CORAAL is not set in db.sh."
  exit 2
fi

log "Download requirements"
pip3 install -r local/requirements.txt

log "Download data to ${CORAAL}"
if [ ! -d "${CORAAL}" ]; then
    mkdir -p "${CORAAL}"
fi
# To absolute path
CORAAL=$(cd ${CORAAL}; pwd)

echo local/download_and_untar.sh ${download_opt} "${CORAAL}" "${data_url}"
local/download_and_untar.sh ${download_opt} "${CORAAL}" "${data_url}"

coraal_audio_dir=${CORAAL}
coraal_text=${CORAAL}/transcript.tsv


log "Data Preparation"

log "Generate segments and transcript"
# generates $coraal_text and segments
echo python3 local/snippet_generation.py "${CORAAL}" "${CORAAL}" 0.1 30
python3 local/snippet_generation.py "${CORAAL}" "${CORAAL}" 0.1 30

log "Text normalization"
mv "${coraal_text}" "${coraal_text}".bak
echo python3 local/text_normalization.py "${coraal_text}".bak "${coraal_text}"
python3 local/text_normalization.py "${coraal_text}".bak "${coraal_text}"

log "Generate train/dev/test splits"
# generates downloads/{train,dev,test}_{wav,utt}.list, which contains the utterances or the wav files for that split
# also generates downloads/{train,dev,test}_{segments,utt2spk}
echo python3 local/train_dev_test_split.py "${coraal_text}" "${CORAAL}"/train "${CORAAL}"/dev "${CORAAL}"/test 0.8 0.1 0.1
python3 local/train_dev_test_split.py "${coraal_text}" "${CORAAL}"/train "${CORAAL}"/dev "${CORAAL}"/test 0.8 0.1 0.1

train_dir=data/local/train
dev_dir=data/local/dev
test_dir=data/local/test

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir

n=$(find $coraal_audio_dir -iname "*.wav" | wc -l)
[ $n -ne 271 ] && \
  log Warning: expected 271 data data files, found $n


# wav.list (<wav_id>)
# note: 1 wav contains many utterances cuz CORAAL is long form
cp ${CORAAL}/train_wav.list $train_dir/wav.list
cp ${CORAAL}/dev_wav.list $dev_dir/wav.list
cp ${CORAAL}/test_wav.list $test_dir/wav.list

# utt.list (<utterance_id>)
cp ${CORAAL}/train_utt.list $train_dir/utt.list
cp ${CORAAL}/dev_utt.list $dev_dir/utt.list
cp ${CORAAL}/test_utt.list $test_dir/utt.list

# utt2spk (<utterance_id> <speaker_id>)
cp ${CORAAL}/train.utt2spk $train_dir/utt2spk
cp ${CORAAL}/dev.utt2spk $dev_dir/utt2spk
cp ${CORAAL}/test.utt2spk $test_dir/utt2spk

# segments (<utterance_id> <wav_id> <start_time> <end_time>)
cp ${CORAAL}/train.segments $train_dir/segments
cp ${CORAAL}/dev.segments $dev_dir/segments
cp ${CORAAL}/test.segments $test_dir/segments

# text (<utterance_id> <transcription>)
cp ${CORAAL}/train.text $train_dir/text
cp ${CORAAL}/dev.text $dev_dir/text
cp ${CORAAL}/test.text $test_dir/text


for dir in $train_dir $dev_dir $test_dir; do
  log "Preparing ${dir} transcriptions"

  # wav.scp (<wav_id> <wav_path>)
    # not utterance_id since we're using the segments format
    # the path (downloads/PRV_se0_ag1_m_01_1.wav)
  cat $dir/wav.list | sed 's/\(.*\)/\1 \1/' | sed 's/\s/ downloads\//g' | sed 's/$/.wav/g' > $dir/wav.scp

  # remove double quotes from text
  mv $dir/text $dir/text.tmp
  cat $dir/text.tmp | sed 's/\"//g' > $dir/text
  rm $dir/text.tmp

  # ensure sorted
  mv $dir/utt2spk $dir/utt2spk.tmp
  mv $dir/wav.scp $dir/wav.scp.tmp
  mv $dir/text $dir/text.tmp
  mv $dir/segments $dir/segments.tmp
  sort -u $dir/utt2spk.tmp > $dir/utt2spk
  sort -u $dir/wav.scp.tmp > $dir/wav.scp
  sort -u $dir/text.tmp > $dir/text
  sort -u $dir/segments.tmp > $dir/segments
  rm $dir/utt2spk.tmp $dir/wav.scp.tmp $dir/text.tmp $dir/segments.tmp

  # spk2utt (<speaker_id> <utterance_id>)
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p data/train data/dev data/test

for f in spk2utt utt2spk wav.scp text segments; do
  cp $train_dir/$f data/train/$f || exit 1;
  cp $dev_dir/$f data/dev/$f || exit 1;
  cp $test_dir/$f data/test/$f || exit 1;
done

for dir in data/train data/dev data/test; do
  utils/validate_data_dir.sh --no-feats $dir || exit 1
done

log "Successfully finished. [elapsed=${SECONDS}s]"
