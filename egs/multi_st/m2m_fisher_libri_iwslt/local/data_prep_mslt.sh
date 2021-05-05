#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <set>"
  echo "e.g.: $0 dat_dir dev"
  exit 1
fi

src=$1
set=$(echo $2 | awk -F"/" '{print $NF}')
if [ ${set} = dev ]; then
   set=MSLT_Dev_EN_20160616
elif [ ${set} = test ]; then
   set=MSLT_Test_EN_20160516
else
  echo "$2 must be dev or test." && exit 1;
fi
src=${src}/${set}
dst=$(pwd)/data/local/${set}

mkdir -p $dst || exit 1;

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1;

wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text.*; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk

# error check
n_wav=$(find -L $src -iname "*.wav" | wc -l)
n_en=$(find -L $src -iname "*.T2.en.snt" | wc -l)
n_fr=$(find -L $src -iname "*.T3.fr.snt" | wc -l)
n_de=$(find -L $src -iname "*.T3.de.snt" | wc -l)
[ $n_wav -ne $n_en ] && echo "Warning: expected $n_wav data data files, found $n_en" && exit 1;
[ $n_wav -ne $n_fr ] && echo "Warning: expected $n_wav data data files, found $n_fr" && exit 1;
[ $n_wav -ne $n_de ] && echo "Warning: expected $n_wav data data files, found $n_de" && exit 1;

# extract meta data
find -L $src -iname "*.wav" | sort | while read line; do
    file_name=$(basename $line | cut -f 1 -d ".")
    echo $file_name | awk -v "path=$line" '{printf "%s cat %s |\n", $0, path}' >> $wav_scp || exit 1;

    # NOTE: assume each utterance is spoken by an independent speaker
    echo $file_name | awk -v "speaker=$file_name" '{printf "%s %s\n", $1, speaker}' >> $utt2spk || exit 1;
done

for lang in en fr de; do
    find -L $src -iname "*.${lang}.snt" | grep -v T1 | sort | while read line; do
        cat $line | tr '\n' ' ' | nkf -Z >> $dst/${lang}.org
        echo -e "" >> $dst/${lang}.org
    done

    # normalize punctuation
    cat $dst/${lang}.org | normalize-punctuation.perl -l ${lang} | local/normalize_punctuation.pl > $dst/${lang}.norm

    # lowercasing
    lowercase.perl < $dst/${lang}.norm > $dst/${lang}.norm.lc
    cp $dst/${lang}.norm $dst/${lang}.norm.tc

    # remove punctuation (not used)
    cat $dst/${lang}.norm.lc | local/remove_punctuation.pl > $dst/${lang}.norm.lc.rm

    # tokenization
    tokenizer.perl -a -l ${lang} -q < $dst/${lang}.norm.tc > $dst/${lang}.norm.tc.tok
    tokenizer.perl -a -l ${lang} -q < $dst/${lang}.norm.lc > $dst/${lang}.norm.lc.tok
    tokenizer.perl -a -l ${lang} -q < $dst/${lang}.norm.lc.rm > $dst/${lang}.norm.lc.rm.tok

    paste -d " " <(awk '{print $1}' $wav_scp) <(cat $dst/${lang}.norm.tc.tok) > $dst/text.tc.${lang}
    paste -d " " <(awk '{print $1}' $wav_scp) <(cat $dst/${lang}.norm.lc.tok) > $dst/text.lc.${lang}
    paste -d " " <(awk '{print $1}' $wav_scp) <(cat $dst/${lang}.norm.lc.rm.tok) > $dst/text.lc.rm.${lang}

    # save original and cleaned punctuation
    cat $dst/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
      | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > $dst/punctuation.${lang}
    cat $dst/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
      | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > $dst/punctuation.clean.${lang}
done


spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

utils/fix_data_dir.sh --utt_extra_files "text.tc.en text.tc.fr text.tc.de \
                                         text.lc.en text.lc.fr text.lc.de" $dst || exit 1;
utils/validate_data_dir.sh --no-feats --no-text $dst || exit 1;


# error check
for f in text.tc.en text.tc.fr text.tc.de; do
    ntrans=$(wc -l <$dst/$f)
    nutt2spk=$(wc -l <$utt2spk)
    ! [ "$ntrans" -eq "$nutt2spk" ] && \
      echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1;
done


# Copy stuff intoc its final locations [this has been moved from the format_data script]
mkdir -p data/$set
for f in spk2utt utt2spk wav.scp; do
    cp $dst/$f data/$set/$f
done
# en
cp $dst/text.tc.en data/$set/text.tc.en
cp $dst/text.lc.en data/$set/text.lc.en
# fr
cp $dst/text.tc.fr data/$set/text.tc.fr
cp $dst/text.lc.fr data/$set/text.lc.fr
# de
cp $dst/text.tc.de data/$set/text.tc.de
cp $dst/text.lc.de data/$set/text.lc.de

echo "$0: successfully prepared data in $dst"
exit 0;
