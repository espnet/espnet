#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
help_message=$(cat << EOF
Usage: $0 [trans_type]

Options:
    trans_type (str): "char" or "phn"
EOF
)
SECONDS=0


# Data preparation related
trans_type="$1"

log "$0 $*"

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if [ -z "${TIMIT}" ]; then
    log "Error: \$TIMIT is not set in db.sh."
    exit 2
fi

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

if [ -z "$trans_type" ]; then
    if [[ "$trans_type" != "char" && "$trans_type" != "phn" ]]; then
        log "Transcript type must be one of [phn, char]"
        log $2
    fi
else
    trans_type=phn
fi

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
    log "Could not find (or execute) the sph2pipe program at $sph2pipe";
    exit 1;
fi

log "Data Preparation"
[ -f $conf/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";

# First check if the train & test directories exist (these can either be upper-
# or lower-cased
if [ ! -d ${TIMIT}/TRAIN -o ! -d ${TIMIT}/TEST ] && [ ! -d ${TIMIT}/train -o ! -d ${TIMIT}/test ]; then
    log "Spot check of command line argument failed"
    log "Command line argument must be absolute pathname to TIMIT directory"
    log "with name like /export/corpora5/LDC/LDC93S1/timit/TIMIT"
    exit 1;
fi

# Now check what case the directory structure is
uppercased=false
train_dir=train
test_dir=test
if [ -d ${TIMIT}/TRAIN ]; then
    uppercased=true
    train_dir=TRAIN
    test_dir=TEST
fi

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 24-speaker core test
# set and the 50-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.
if $uppercased; then
    tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
    tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
    ls -d "${TIMIT}"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
else
    tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
    tr '[:upper:]' '[:lower:]' < $conf/test_spk.list > $tmpdir/test_spk
    ls -d "${TIMIT}"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
fi

cd $dir
for x in train dev test; do
    # First, find the list of audio files (use only si & sx utterances).
    # Note: train & test sets are under different directories, but doing find on
    # both and grepping for the speakers will work correctly.
    find ${TIMIT}/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $tmpdir/${x}_spk > ${x}_sph.flist
    
    sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' ${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
    paste $tmpdir/${x}_sph.uttids ${x}_sph.flist \
    | sort -k1,1 > ${x}_sph.scp
    
    cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids
    
    # Now, Convert the transcripts into our format (no normalization yet)
    # Get the transcripts: each line of the output contains an utterance
    # ID followed by the transcript.
    
    if [ $trans_type = "phn" ]
    then
        log "phone transcript!"
        find ${TIMIT}/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.PHN' \
        | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist
        sed -e 's:.*/\(.*\)/\(.*\).PHN$:\1_\2:i' $tmpdir/${x}_phn.flist \
        > $tmpdir/${x}_phn.uttids
        while read line; do
            [ -f $line ] || error_exit "Cannot find transcription file '$line'";
            cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
        done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
        paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
        | sort -k1,1 > ${x}.trans
        
    elif [ $trans_type = "char" ]
    then
        log "char transcript!"
        find ${TIMIT}/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WRD' \
        | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_wrd.flist
        sed -e 's:.*/\(.*\)/\(.*\).WRD$:\1_\2:i' $tmpdir/${x}_wrd.flist \
        > $tmpdir/${x}_wrd.uttids
        while read line; do
            [ -f $line ] || error_exit "Cannot find transcription file '$line'";
            cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z  A-Z]//g'
        done < $tmpdir/${x}_wrd.flist > $tmpdir/${x}_wrd.trans
        paste $tmpdir/${x}_wrd.uttids $tmpdir/${x}_wrd.trans \
        | sort -k1,1 > ${x}.trans
    else
        log "WRONG!"
        log $trans_type
        exit 0;
    fi
    
    # Do normalization steps.
    cat ${x}.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map -to 39 | sort > $x.text || exit 1;
    
    # Create wav.scp
    awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp > ${x}_wav.scp
    
    # Make the utt2spk and spk2utt files.
    cut -f1 -d'_'  $x.uttids | paste -d' ' $x.uttids - > $x.utt2spk
    cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
    
    # Prepare gender mapping
    cat $x.spk2utt | awk '{print $1}' | perl -ane 'chop; m:^.:; $g = lc($&); print "$_ $g\n";' > $x.spk2gender
    
    # Prepare STM file for sclite:
    wav-to-duration --read-entire-file=true scp:${x}_wav.scp ark,t:${x}_dur.ark || exit 1
    awk -v dur=${x}_dur.ark \
    'BEGIN{
    while(getline < dur) { durH[$1]=$2; }
    print ";; LABEL \"O\" \"Overall\" \"Overall\"";
    print ";; LABEL \"F\" \"Female\" \"Female speakers\"";
    print ";; LABEL \"M\" \"Male\" \"Male speakers\"";
}
{ wav=$1; spk=wav; sub(/_.*/,"",spk); $1=""; ref=$0;
    gender=(substr(spk,0,1) == "f" ? "F" : "M");
    printf("%s 1 %s 0.0 %f <O,%s> %s\n", wav, spk, durH[wav], gender, ref);
}
    ' ${x}.text >${x}.stm || exit 1
    
    # Create dummy GLM file for sclite:
    log ';; empty.glm
[FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
    ' > ${x}.glm
done

# Copyright 2013  (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

log "Preparing train, dev and test data"
tmpdir=data/local/lm_tmp
lexicon=data/local/dict/lexicon.txt
mkdir -p $tmpdir

for x in train dev test; do
    mkdir -p data/${x}
    cp ${dir}/${x}_wav.scp data/${x}/wav.scp || exit 1;
    cp ${dir}/${x}.text data/${x}/text || exit 1;
    cp ${dir}/${x}.spk2utt data/${x}/spk2utt || exit 1;
    cp ${dir}/${x}.utt2spk data/${x}/utt2spk || exit 1;
    utils/filter_scp.pl data/${x}/spk2utt ${dir}/${x}.spk2gender > data/${x}/spk2gender || exit 1;
    cp ${dir}/${x}.stm data/${x}/stm
    cp ${dir}/${x}.glm data/${x}/glm
    utils/validate_data_dir.sh --no-feats data/${x} || exit 1
done


log "Successfully finished. [elapsed=${SECONDS}s]"
