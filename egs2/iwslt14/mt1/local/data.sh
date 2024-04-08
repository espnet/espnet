#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh || exit 1;
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000
URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${IWSLT14}" ]; then
    log "Fill the value of 'IWSLT14' of db.sh"
    exit 1
fi


if [ -f "${IWSLT14}/${GZ}" ]; then
    log "Data already downloaded"
else
    (
        cd ${IWSLT14}
        wget "$URL"
        tar zxvf $GZ
    )
    log "Data downloaded and extracted"
fi

src=de
tgt=en
lang=de-en
prep=iwslt14.tokenized.de-en
tmp=data/$prep/tmp

if [ ! -d "${IWSLT14}/${lang}" ]; then
    (
        cd ${IWSLT14}
        tar zxvf $GZ
    )
    log "Data extracted"
fi

# check extra module installation
if ! command -v tokenizer.perl > /dev/null; then
    echo "Error: it seems that moses is not installed." >&2
    echo "Error: please install moses as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make moses.done" >&2
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"
    mkdir -p data/train data/valid data/test $tmp


    log "preparing test and valid data"

    for l in $src $tgt; do
        for o in "${IWSLT14}/${lang}"/IWSLT14.TED*."${l}".xml; do
            fname=${o##*/}
            f=$tmp/${fname%.*}
            echo $o $f
            grep '<seg id' $o | \
                sed -e 's/<seg id="[0-9]*">\s*//g' | \
                sed -e 's/\s*<\/seg>\s*//g' | \
                sed -e "s/\’/\'/g" > $f
            tokenizer.perl -threads 8 -l $l < $f > $f.tok
            lowercase.perl < $f.tok > $f.tok.lc
            remove_punctuation.pl < $f.tok > $f.tok.rm
            remove_punctuation.pl < $f.tok.lc > $f.tok.lc.rm
            echo ""
        done
    done

    log "pre-processing train data..."
    for l in $src $tgt; do
        f=train.tags.$lang.$l
        tok=train.tags.$lang.tok.$l

        < $IWSLT14/$lang/$f \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        sed -e 's/<title>//g' | \
        sed -e 's/<\/title>//g' | \
        sed -e 's/<description>//g' | \
        sed -e 's/<\/description>//g' > $tmp/$f
        tokenizer.perl -threads 8 -l $l < $tmp/$f > $tmp/$tok
        echo ""
    done

    log "Cleaning train data"
    clean-corpus-n.perl -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.tok.clean 1 175
    for l in $src $tgt; do
        lowercase.perl < $tmp/train.tags.$lang.tok.clean.$l > $tmp/train.tags.$lang.tok.clean.lc.$l

        remove_punctuation.pl < $tmp/train.tags.$lang.tok.clean.lc.$l > $tmp/train.tags.$lang.tok.lc.rm.$l
        remove_punctuation.pl < $tmp/train.tags.$lang.tok.clean.$l > $tmp/train.tags.$lang.tok.rm.$l
    done

    #Clean again
    clean-corpus-n.perl $tmp/train.tags.$lang.tok.lc.rm $src $tgt $tmp/train.tags.$lang.tok.clean.lc.rm 1 175
    clean-corpus-n.perl $tmp/train.tags.$lang.tok.rm $src $tgt $tmp/train.tags.$lang.tok.clean.rm 1 175

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Creating Splits"
    for l in $src $tgt; do

        awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.$l > $tmp/train.tok.clean.$l
        awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.$l > $tmp/valid.tok.clean.$l

        awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.lc.$l > $tmp/train.tok.clean.lc.$l
        awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.lc.$l > $tmp/valid.tok.clean.lc.$l

        awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.lc.rm.$l > $tmp/train.tok.clean.lc.rm.$l
        awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.lc.rm.$l > $tmp/valid.tok.clean.lc.rm.$l

        awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.rm.$l > $tmp/train.tok.clean.rm.$l
        awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$lang.tok.clean.rm.$l > $tmp/valid.tok.clean.rm.$l

        cat $tmp/IWSLT14.TED.dev2010.$lang.$l.tok \
            $tmp/IWSLT14.TEDX.dev2012.$lang.$l.tok \
            $tmp/IWSLT14.TED.tst2010.$lang.$l.tok \
            $tmp/IWSLT14.TED.tst2011.$lang.$l.tok \
            $tmp/IWSLT14.TED.tst2012.$lang.$l.tok \
            > $tmp/test.tok.$l

        cat $tmp/IWSLT14.TED.dev2010.$lang.$l.tok.lc \
            $tmp/IWSLT14.TEDX.dev2012.$lang.$l.tok.lc \
            $tmp/IWSLT14.TED.tst2010.$lang.$l.tok.lc \
            $tmp/IWSLT14.TED.tst2011.$lang.$l.tok.lc \
            $tmp/IWSLT14.TED.tst2012.$lang.$l.tok.lc \
            > $tmp/test.tok.lc.$l

        cat $tmp/IWSLT14.TED.dev2010.$lang.$l.tok.lc.rm \
            $tmp/IWSLT14.TEDX.dev2012.$lang.$l.tok.lc.rm \
            $tmp/IWSLT14.TED.tst2010.$lang.$l.tok.lc.rm \
            $tmp/IWSLT14.TED.tst2011.$lang.$l.tok.lc.rm \
            $tmp/IWSLT14.TED.tst2012.$lang.$l.tok.lc.rm \
            > $tmp/test.tok.lc.rm.$l

        cat $tmp/IWSLT14.TED.dev2010.$lang.$l.tok.rm \
            $tmp/IWSLT14.TEDX.dev2012.$lang.$l.tok.rm \
            $tmp/IWSLT14.TED.tst2010.$lang.$l.tok.rm \
            $tmp/IWSLT14.TED.tst2011.$lang.$l.tok.rm \
            $tmp/IWSLT14.TED.tst2012.$lang.$l.tok.rm \
            > $tmp/test.tok.rm.$l

        nl -s ' ' -n rz $tmp/train.tok.clean.$l | awk '{print "utt" $0}' > data/train/text.tc.$l
        nl -s ' ' -n rz $tmp/train.tok.clean.rm.$l | awk '{print "utt" $0}' > data/train/text.tc.rm.$l
        nl -s ' ' -n rz $tmp/train.tok.clean.lc.$l | awk '{print "utt" $0}' > data/train/text.lc.$l
        nl -s ' ' -n rz $tmp/train.tok.clean.lc.rm.$l | awk '{print "utt" $0}' > data/train/text.lc.rm.$l

        nl -s ' ' -n rz $tmp/valid.tok.clean.$l | awk '{print "utt" $0}' > data/valid/text.tc.$l
        nl -s ' ' -n rz $tmp/valid.tok.clean.rm.$l | awk '{print "utt" $0}' > data/valid/text.tc.rm.$l
        nl -s ' ' -n rz $tmp/valid.tok.clean.lc.$l | awk '{print "utt" $0}' > data/valid/text.lc.$l
        nl -s ' ' -n rz $tmp/valid.tok.clean.lc.rm.$l | awk '{print "utt" $0}' > data/valid/text.lc.rm.$l

        nl -s ' ' -n rz $tmp/test.tok.$l | awk '{print "utt" $0}' > data/test/text.tc.$l
        nl -s ' ' -n rz $tmp/test.tok.rm.$l | awk '{print "utt" $0}' > data/test/text.tc.rm.$l
        nl -s ' ' -n rz $tmp/test.tok.lc.$l | awk '{print "utt" $0}' > data/test/text.lc.$l
        nl -s ' ' -n rz $tmp/test.tok.lc.rm.$l | awk '{print "utt" $0}' > data/test/text.lc.rm.$l

    done
fi
