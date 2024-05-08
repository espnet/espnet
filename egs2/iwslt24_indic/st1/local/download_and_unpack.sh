#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 3 ]; then
    echo "Usage: $0 <dataset dir> <tgt_lang> <remove_archive>"
    echo "e.g.: $0 /path/to/indic/data hi true"
fi

data_dir=$1
tgt_lang=$2
remove_archive=$3

# check tgt_lang
if [ "$tgt_lang" == "hi" ]; then
    target_language="Hindi"
elif [ "$tgt_lang" == "bn" ]; then
    target_language="Bengali"
elif [ "$tgt_lang" == "ta" ]; then
    target_language="Tamil"
else
    log "Error: ${tgt_lang} is not supported. It must be one of hi, bn, or ta."
    exit 1;
fi

# check if the dataset is already unpacked
if [ -f ${data_dir}/.unpacked_en-${tgt_lang} ]; then
    log "$0: Data has already been extracted successfully. Skipping this stage."
    exit 0
fi

# check if zip files are present
if [ ! -f "${data_dir}/${target_language}.zip" ]; then
    log "Please contact IWSLT 2024 Indic track organizers (See project website: https://iwslt.org/2024/indic) to download the training and development data, and place the zip files inside ${data_dir}."
fi
if [ ! -f "${data_dir}/${target_language}-Test.zip" ]; then
    log "Please contact IWSLT 2024 Indic track organizers (See project website: https://iwslt.org/2024/indic) to download the test data, and place the zip files inside ${data_dir}."
fi

# unzip files
if [ ! -d "${data_dir}/${target_language}" ]; then
    UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip "${data_dir}/${target_language}.zip" -d ${data_dir}
fi
if [ ! -d "${data_dir}/${target_language}-Test" ]; then
    unzip "${data_dir}/${target_language}-Test.zip" -d ${data_dir}
fi

# reorganize directories
mv ${data_dir}/${target_language}/en-${tgt_lang} ${data_dir}/
mv ${data_dir}/${target_language}-Test/tst-COMMON ${data_dir}/en-${tgt_lang}/data/
rmdir ${data_dir}/${target_language}
rmdir ${data_dir}/${target_language}-Test

# remove zip files if necessary
if ${remove_archive}; then
    log "Removing zip files..."
    rm ${data_dir}/${target_language}.zip
    rm ${data_dir}/${target_language}-Test.zip
fi

touch ${data_dir}/.unpacked_en-${tgt_lang}
log "$0: Successfully downloaded and unpacked en-${tgt_lang}"
