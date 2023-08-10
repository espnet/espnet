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
stop_stage=4
data_url=www.openslr.org/resources/12
train_dev="dev"
asr_data_dir=
asr_stats_dir=
files=
bpemodel=

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${LIBRISPEECH}/LibriSpeech/LICENSE.TXT" ]; then
	echo "stage 1: Data Download to ${LIBRISPEECH}"
	for part in dev-clean test-clean dev-other test-other train-clean-100; do
            local/download_and_untar.sh ${LIBRISPEECH} ${data_url} ${part}
	done
    else
        log "stage 1: ${LIBRISPEECH}/LibriSpeech/LICENSE.TXT is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${LIBRISPEECH}/LibriSpeech/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: combine all training and development sets"
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} data/dev_clean data/dev_other
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # use external data
    if [ ! -e data/local/other_text/librispeech-lm-norm.txt.gz ]; then
	log "stage 4: prepare external text data from http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/other_text/
    fi
    if [ ! -e data/local/other_text/text ]; then
	# provide utterance id to each texts
	# e.g., librispeech_lng_00003686 A BANK CHECK
	zcat data/local/other_text/librispeech-lm-norm.txt.gz | \
	    awk '{ printf("librispeech_lng_%08d %s\n",NR,$0) } ' > data/local/other_text/text
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      log "stage 5: prepare external text data from train-clean-360 train-other-500"
      for part in train-clean-360 train-other-500; do
	  if [ ! -e "${LIBRISPEECH}/LibriSpeech/${part}" ]; then
	      local/download_and_untar.sh ${LIBRISPEECH} ${data_url} ${part}
	      local/data_prep.sh ${LIBRISPEECH}/LibriSpeech/${part} data/${part//-/_}
	    else
	      log "stage 5: ${LIBRISPEECH}/LibriSpeech/${part} is already existing Skip data downloading"
	  fi
      done
      if [ ! -e "data/local/860_text/text" ]; then
	  mkdir "data/local/860_text/"

	  for part in train_clean_360 train_other_500; do
	    cat "data/${part}/text" >> "data/local/860_text/text"

	  done
      else
	  log "stage 5: data/local/860_text/text is already existing Skip it"
      fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      log "stage 6: combine training data and external text data"
      sed -e 's/^/external-/' data/local/860_text/text > ${asr_data_dir}/text_injection

      # utt2category
      <${asr_data_dir}/wav.scp awk '{print($1, "SPEECH")}' > ${asr_data_dir}/utt2category
      <${asr_data_dir}/text_injection awk '{print($1, "TEXT_INJECTION")}' >> ${asr_data_dir}/utt2category

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "stage 7: create extra text data's shape information"
    python local/create_extra_info.py --files ${files} \
      --asr_data_dir ${asr_data_dir} \
      --bpe_model ${bpemodel} \
      --asr_stats_dir ${asr_stats_dir}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
