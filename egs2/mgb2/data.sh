#!/usr/bin/env bash
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

langs=
recog= 
. utils/parse_options.sh

for l in ${langs} ${recog}; do
  if [ ! -e "${BABEL}_${l}" ]; then
      log "Fill the value of '${BABEL}_${l}' of db.sh"
      exit 1
  fi
done


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train Directories
train_set=train
train_dev=dev

recog_set=""
for l in ${recog}; do
  recog_set="dev_${l} eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }


./local/setup_languages.sh --langs "${langs}" --recog "${recog}"


# language specific : 

for x in ${train_set} ${train_dev} ${recog_set}; do
  sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
done

cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > data/nlsym.txt

mv data/train data/train_${langs}
mv data/nlsym.txt data/nlsym.${langs}.txt


log "Successfully finished. [elapsed=${SECONDS}s]"
