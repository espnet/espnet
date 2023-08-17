#!/usr/bin/env bash

# Copyright 2021 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

source ./utils/simple_dict.sh

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
lang=all  # if set as specific language, only download that language


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./utils/parse_options.sh

log "data preparation started"

if [ -z "${MLS}" ]; then
    log "Fill the value of 'MLS' of db.sh"
    exit 1
fi

# Languages of MLS
langs=(es en fr nl it pt pl de)
lang_names=(spanish english french dutch italian portuguese polish german)

dict_init lang_dict
for i in {0..7}; do
    if [ "$lang" == "all" ] || [ "$lang" == "${langs[i]}" ]; then
        dict_put lang_dict ${langs[i]} ${lang_names[i]}
    fi
done

# Download and read md5sum.txt
if ! which wget >/dev/null; then
    log "$0: wget is not installed."
    exit 1;
fi
url="https://dl.fbaipublicfiles.com/mls/md5sum.txt"
if ! wget --no-check-certificate $url -P ${MLS}; then
    echo "$0: error executing wget $url"
    exit 1;
fi
dict_init md5_dict
while read line; do
    md5=$( echo $line | cut -d " " -f 1)
    filename=$( echo $line | cut -d " " -f 2)
    dict_put md5_dict $filename $md5
done < "${MLS}/md5sum.txt"
# Finshed: Download and read md5sum.txt

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to ${MLS}"

    for lang_name in $(dict_values lang_dict); do
        data_filename="mls_${lang_name}.tar.gz"
        lm_filename="mls_lm_${lang_name}.tar.gz"

        local/download_and_untar.sh \
            --md5sum $(dict_get md5_dict ${data_filename})\
            "${MLS}"\
            "https://dl.fbaipublicfiles.com/mls/${data_filename}" \
            "${data_filename}"
        local/download_and_untar.sh \
            --md5sum $(dict_get md5_dict ${lang_filename})\
            "${MLS}"\
            "https://dl.fbaipublicfiles.com/mls/${lm_filename}" \
            "${lm_filename}"
        # Optional: mls corpus is large. You might want to remove them after processing by supplying --remove-archive.
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing Data for MLS"

    for lang in $(dict_keys lang_dict); do
        lang_name=$(dict_get lang_dict ${lang})

        python local/data_prep.py --source ${MLS}/mls_${lang_name} --lang ${lang} --prefix "mls_"
        for split in train dev test; do
            utils/fix_data_dir.sh data/mls_${lang}_${split}
        done

        # add placeholder to align format with other corpora
        sed -r '/^\s*$/d' ${MLS}/mls_lm_${lang_name}/data.txt | \
            awk '{printf("%.8d %s\n"), NR-1, $0}'  > data/${lang}_lm_train.txt
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
