#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# General configuration
stage=0
stop_stage=100
nj=12
corpus_dir=data/local/corpus  # TODO(fhrozen): remove maybe
corpus_txt=data/train/text  # TODO(fhrozen): remove maybe
workdir=data/local/mfa  
clean_temp=false
datasets="tr_no_dev dev eval1"  # TODO(fhrozen): use this as main

# Feature extraction related
fs=16000

# Tokenization related
lang="english_us_tacotron"
g2p="espeak_ng_english_us_vits"
train=true
max_phonemes_word=7

help_message=$(cat << EOF
Usage: $0 --stage "<stage>" --stop-stage "<stop_stage>" --fs "<fs>"

Options:
    # General configuration
    --stage                # Processes starts from the specified stage (default="${stage}").
    --stop_stage            # Processes is stopped at the specified stage (default="${stop_stage}").
    --nj                    # The number of parallel jobs (default="${nj}").
    --workdir
    --clean_temp
    --datasets

    # Feature extraction related
    --fs                      # Sampling rate (default="${fs}").

    # Tokenization related
    --lang
    --g2p
    --train
    --max_phonemes_word

EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh
echo $@
if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

mkdir -p ${corpus_dir}  # TODO(fhrozen): fix this hardcoded value
mkdir -p ${workdir}
tempdir=${workdir}/tmp

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Prepare Data set for MFA"
    # TODO(fhrozen): replace this for a general data set, and use utt2spk to generate the folder structure
    for fn in $(find ${db_root}/LJSpeech-1.1/wavs -name "*.wav");do
        ln -sf $(readlink -e ${fn}) ${corpus_dir}
    done

    # Text cleaning and save it in lab files (independly)
    python pyscripts/prepare_labs.py --cleaner tacotron \
        ${corpus_txt} \
        ${corpus_dir}
    
    cat ${corpus_dir}/*.lab | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' > ${workdir}/list.txt.tmp
    paste ${workdir}/list.txt.tmp ${workdir}/list.txt.tmp > ${workdir}/list2.txt.tmp

    # Generate a text using espnet2-based g2p
    python pyscripts/mfa/reformat_dict.py \
            --g2p ${g2p} \
            ${workdir}/list2.txt.tmp
    
    mv ${workdir}/modified_list2.txt.tmp ${workdir}/train_dict.txt
    rm ${workdir}/*.tmp
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if ${train}; then
        log "stage 1: Training MFA"

        # Complete training
        mfa train_g2p --clean\
            --phonetisaurus \
            -t ${tempdir} \
            ${workdir}/train_dict.txt ${workdir}/${lang}.zip

        mfa g2p --clean -t ${tempdir} \
            ${workdir}/${lang}.zip \
            ${corpus_dir} \
            ${workdir}/lexicon.txt

        mfa train -t ${tempdir} \
            ${corpus_dir} \
            ${workdir}/lexicon.txt \
            ${workdir}/acoustic_model.zip
    else
        log "stage 1: Downloading required files for MFA"
        # TODO (fix this lexicons for general)
        wget -q -nc --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P data/mfa_local
        wget -q -nc --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P data/mfa_local
        python pyscripts/mfa/reformat_dict.py \
            --g2p espeak_ng_english_us_vits \
            --cleaner tacotron \
            data/mfa_local/librispeech-lexicon.txt || exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Generating aligments using MFA model"

    if ${train}; then
        lexicon=${workdir}/lexicon.txt
        acoustic=${workdir}/acoustic_model.zip
    else
        lexicon=data/mfa_local/modified_librispeech-lexicon.txt
        acoustic=data/mfa_local/english.zip

        mfa validate --clean -j ${nj} \
        -t ${tempdir} \
        ${corpus_dir} \
        ${lexicon} \
        ${acoustic}
    fi

# Remove punctuation and clitic from aligment, otherwise it will generate a issue with g2p model
cat << EOF > "${workdir}"/config.yaml
punctuation: null
clitic_markers: null
EOF

    mfa align -j ${nj} \
        --clean \
        -t ${tempdir} \
        --config_path "${workdir}"/config.yaml \
        --output_format json \
        ${corpus_dir} \
        ${lexicon} \
        ${acoustic} \
        ${workdir}/aligments

    if ${clean}; then
        # Be careful, this will delete all the files employed for training the mfa's models.
        rm -rf ${tempdir}
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare phoneme-text labels"

    echo "<sil>" > data/local/nlsyms.txt

    python local/get_phones_alignments.py \
        --samplerate ${fs} \
        --g2p ${g2p} \
        --max_phonemes_word ${max_phonemes_word} \
        ${workdir}/aligments \
        ${workdir}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Prepare Data Sets"
    for dset in ${datasets}; do
        utils/copy_data_dir.sh data/"${dset}"{,_phn}
        cp ${workdir}/text.phn data/${dset}_phn/text
        utils/fix_data_dir.sh data/${dset}_phn

        utils/filter_scp.pl data/${dset}_phn/utt2spk ${workdir}/durations > data/${dset}_phn/durations
        utils/filter_scp.pl data/${dset}_phn/utt2spk ${workdir}/word_durations > data/${dset}_phn/word_durations
    done
fi
