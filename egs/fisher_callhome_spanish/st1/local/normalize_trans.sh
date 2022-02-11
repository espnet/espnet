#!/usr/bin/env bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <LDC2010T04-location> <LDC96T17-location>"
    echo "e.g.: $0 /export/corpora/LDC/LDC2010T04 /export/corpora/LDC/LDC96T17"
    exit 1;
fi

# download data preparation scripts for transcriptions
# Note uses fork because of utf-8 issues in iconv - https://github.com/joshua-decoder/fisher-callhome-corpus/pull/3
[ ! -d data/local/fisher-callhome-corpus ] && git clone https://github.com/siddalmia/fisher-callhome-corpus.git data/local/fisher-callhome-corpus

# create symbolic links
cur_dir=$(pwd)
cd local/
rm -rf mapping
ln -s $cur_dir/data/local/fisher-callhome-corpus/mapping .
cd $cur_dir

# data preparation for Es
cd data/local/fisher-callhome-corpus
./bin/build_fisher.sh $1/fisher_spa_tr
./bin/build_callhome.sh $2/callhome_spanish_trans_970711
cd $cur_dir

# example
# orignial:
#     So, here, we've been this year. And later, I'm going home, in Barranquilla. Eh, next, next year, God willing.
# truecase + tokenization (tc):
#     So , here , we &apos;ve been this year . And later , I &apos;m going home , in Barranquilla . Eh , next , next year , God willing .
# lowercase + tokenization (lc):
#     so , here , we &apos;ve been this year. and later , i &apos;m going home , in barranquilla. eh , next , next year , god willing .
# lowercase w/o punctuation + tokenization (lc.rm):
#     so here we &apos;ve been this year and later i &apos;m going home in barranquilla eh next next year god willing

for set in fisher_train fisher_dev fisher_dev2 fisher_test callhome_train callhome_devtest callhome_evltest; do
    # concatenate short utterances
    cp data/${set}/text data/${set}/text.tmp
    cp data/${set}/segments data/${set}/segments.tmp
    local/concat_short_utt.py local/mapping/${set} data/${set}/text.tmp data/${set}/segments.tmp
    rm data/${set}/*.tmp

    # fix spk2utt and utt2spk
    ! perl -ane 'm:([^-]+)-([AB])-(\S+)-(\S+): || die "Bad line $_;"; if ($3 != $4) {print "$1-$2-$3-$4 $1-$2\n"; }' data/${set}/text > data/${set}/utt2spk \
        && echo "Error producing utt2spk file" && exit 1;
    # NOTE: remove inappropriate segmentation (start and end points are the same)
    utils/utt2spk_to_spk2utt.pl < data/${set}/utt2spk > data/${set}/spk2utt

    # normalize punctuation
    cp data/local/fisher-callhome-corpus/corpus/ldc/${set}.es data/${set}/es.joshua.org
    normalize-punctuation.perl -l es < data/${set}/es.joshua.org  | local/normalize_punctuation.pl > data/${set}/es.joshua.norm.tc
    lowercase.perl < data/${set}/es.joshua.norm.tc > data/${set}/es.joshua.norm.lc
    remove_punctuation.pl < data/${set}/es.joshua.norm.lc > data/${set}/es.joshua.norm.lc.rm
    for case in lc.rm lc tc; do
        tokenizer.perl -l es -q < data/${set}/es.joshua.norm.${case} > data/${set}/es.joshua.norm.${case}.tok
    done

    # Now checking these Es transcriptions are matching (double check)
    cp data/${set}/text data/${set}/text.tmp
    cut -f 2- -d " " data/${set}/text.tmp > data/${set}/es.kaldi.org
    normalize-punctuation.perl -l es < data/${set}/es.kaldi.org | local/normalize_punctuation.pl > data/${set}/es.kaldi.norm.tc
    lowercase.perl < data/${set}/es.kaldi.norm.tc > data/${set}/es.kaldi.norm.lc
    tokenizer.perl -l es -q < data/${set}/es.kaldi.norm.tc > data/${set}/es.kaldi.norm.tc.tok
    tokenizer.perl -l es -q < data/${set}/es.kaldi.norm.lc > data/${set}/es.kaldi.norm.lc.tok

    # use references from joshua-decoder/fisher-callhome-corpus
    for case in lc.rm lc tc; do
        paste -d " " <(awk '{print $1}' data/${set}/text.tmp) <(cat data/${set}/es.joshua.norm.${case}.tok | awk '{if(NF>0) {print $0;} else {print "emptyutterance";}}') \
            > data/${set}/text.${case}.es
    done

    # save original and cleaned punctuation
    text2token.py -s 0 -n 1 data/${set}/es.joshua.org | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > data/${set}/punctuation.es
    text2token.py -s 0 -n 1 data/${set}/es.joshua.norm.tc | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > data/${set}/punctuation.clean.es
done


# prepare En references
for set in fisher_train callhome_train callhome_devtest callhome_evltest; do
    # having one En reference
    cp data/local/fisher-callhome-corpus/corpus/ldc/${set}.en data/${set}/en.org
    normalize-punctuation.perl -l en < data/${set}/en.org | sed -e "s/¿//g" | local/normalize_punctuation.pl > data/${set}/en.norm.tc
    lowercase.perl < data/${set}/en.norm.tc > data/${set}/en.norm.lc
    remove_punctuation.pl < data/${set}/en.norm.lc > data/${set}/en.norm.lc.rm
    for case in lc.rm lc tc; do
        tokenizer.perl -l en -q < data/${set}/en.norm.${case} > data/${set}/en.norm.${case}.tok
        paste -d " " <(awk '{print $1}' data/${set}/text.${case}.es) <(cat data/${set}/en.norm.${case}.tok | awk '{if(NF>0) {print $0;} else {print "emptyutterance";}}') \
            > data/${set}/text.${case}.en
    done

    # save original and cleaned punctuation
    text2token.py -s 0 -n 1 data/${set}/en.org | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > data/${set}/punctuation.en
    text2token.py -s 0 -n 1 data/${set}/en.norm.tc | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > data/${set}/punctuation.clean.en
done
for set in fisher_dev fisher_dev2 fisher_test; do
    # having four En references
    for no in 0 1 2 3; do
        cp data/local/fisher-callhome-corpus/corpus/ldc/${set}.en.${no} data/${set}/en.${no}.org
        normalize-punctuation.perl -l en < data/${set}/en.${no}.org | sed -e "s/¿//g"| local/normalize_punctuation.pl > data/${set}/en.${no}.norm.tc
        lowercase.perl < data/${set}/en.${no}.norm.tc > data/${set}/en.${no}.norm.lc
        remove_punctuation.pl < data/${set}/en.${no}.norm.lc > data/${set}/en.${no}.norm.lc.rm
        for case in lc.rm lc tc; do
            tokenizer.perl -l en -q < data/${set}/en.${no}.norm.${case} > data/${set}/en.${no}.norm.${case}.tok
            paste -d " " <(awk '{print $1}' data/${set}/text.${case}.es) <(cat data/${set}/en.${no}.norm.${case}.tok | awk '{if(NF>0) {print $0;} else {print "emptyutterance";}}') \
                > data/${set}/text.${case}.en.${no}
        done
    done

    # save original and cleaned punctuation
    cat data/${set}/en.*.org | text2token.py -s 0 -n 1 | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > data/${set}/punctuation.en
    cat data/${set}/en.*.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > data/${set}/punctuation.clean.en
done


# error check
for set in fisher_train callhome_train callhome_devtest callhome_evltest; do
    n_es=$(cat data/${set}/text.tc.es | wc -l)
    n_en=$(cat data/${set}/text.tc.en | wc -l)
    [ ${n_es} -ne ${n_en} ] && echo "Warning: expected ${n_es} data files, found ${n_en}" && exit 1;
done
for set in fisher_dev fisher_dev2 fisher_test; do
    n_es=$(cat data/${set}/text.tc.es | wc -l)
    for no in 0 1 2 3; do
        n_en=$(cat data/${set}/text.tc.en.${no} | wc -l)
        [ ${n_es} -ne ${n_en} ] && echo "Warning: expected ${n_es} data files, found ${n_en}" && exit 1;
    done
done
# NOTE: the number of orignial utterances is as follow:
# fisher_train: 138819
# fisher_dev: 3979
# fisher_dev2: 3961
# fisher_test: 3641
# callhome_train: 15080
# callhome_devtest: 3966
# callhome_evltest: 1829


# remove empty and short utterances
for set in fisher_train fisher_dev fisher_dev2 fisher_test callhome_train callhome_devtest callhome_evltest; do
    cp -rf data/${set} data/${set}.tmp
    grep -v emptyutterance data/${set}/text.tc.es | cut -f 1 -d " " | sort > data/${set}/reclist.es
    if [ -f data/${set}/text.tc.en ]; then
        grep -v emptyutterance data/${set}/text.tc.en | cut -f 1 -d " " | sort > data/${set}/reclist.en
    else
        for no in 0 1 2 3; do
            grep -v emptyutterance data/${set}/text.tc.en.${no} | cut -f 1 -d " " | sort > data/${set}/reclist.en.${no}
        done
        comm -12 data/${set}/reclist.en.0 data/${set}/reclist.en.1 > data/${set}/reclist.en
        cp data/${set}/reclist.en data/${set}/reclist.en.tmp
        comm -12 data/${set}/reclist.en.tmp data/${set}/reclist.en.2 > data/${set}/reclist.en
        cp data/${set}/reclist.en data/${set}/reclist.en.tmp
        comm -12 data/${set}/reclist.en.tmp data/${set}/reclist.en.3 > data/${set}/reclist.en
    fi
    comm -12 data/${set}/reclist.es data/${set}/reclist.en > data/${set}/reclist
    reduce_data_dir.sh data/${set}.tmp data/${set}/reclist data/${set}
    if [ -f data/${set}/text.tc.en ]; then
        utils/fix_data_dir.sh --utt_extra_files "text.tc.es text.lc.es text.lc.rm.es \
                                                 text.tc.en text.lc.en text.lc.rm.en" data/${set}
    else
        utils/fix_data_dir.sh --utt_extra_files "text.tc.es text.lc.es text.lc.rm.es \
                                                 text.tc.en.0 text.tc.en.1 text.tc.en.2 text.tc.en.3 \
                                                 text.lc.en.0 text.lc.en.1 text.lc.en.2 text.lc.en.3 \
                                                 text.lc.rm.en.0 text.lc.rm.en.1 text.lc.rm.en.2 text.lc.rm.en.3" data/${set}
    fi
    rm -rf data/${set}.tmp
done


echo "$0: successfully prepared data"
exit 0;
