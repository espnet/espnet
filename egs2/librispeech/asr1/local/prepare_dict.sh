#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# General configuration
stage=1
python=python3       # Specify python to execute espnet commands.
ngram_dir=./data/local/lm
asr_train_config=
lang_dir=
bpemodel=

. utils/parse_options.sh

if [ -z $bpemodel ]; then
  bpemodel=$(grep bpemodel $asr_train_config | awk '{print $2}')
  if [ ! -f ${bpemodel} ]; then
    echo "${bpemodel} doesn't exist, check that"
    exit 1
  fi
fi

. ./path.sh

if [ $stage -le 0 ]; then
  local/download_lm.sh "openslr.org/resources/11" ${ngram_dir}
fi

if [ ! -d $lang_dir ]; then
  mkdir -p ${lang_dir}
fi
if [ $stage -le 1 ]; then
  echo "Extract token list from ${asr_train_config}"

  ${python} -m espnet2.bin.extract_token_list \
    --config_file ${asr_train_config} \
    --token_file ${lang_dir}/tokens.txt
fi
if [ $stage -le 2 ]; then
    vocab_file=${ngram_dir}/librispeech-vocab.txt

    paste \
      "${vocab_file}" \
        <(<"${vocab_file}" \
              ${python} -m espnet2.bin.tokenize_text  \
                  --input - --output - \
                  --token_type bpe \
                  --bpemodel "${bpemodel}" \
                ) \
        > ${lang_dir}/lexicon.txt


    echo "<eps> 0" > $lang_dir/words.txt
    echo "<UNK> 1" >> $lang_dir/words.txt
    awk '{print $1, FNR + 1}' ${vocab_file} >> ${lang_dir}/words.txt

    echo "<UNK> <unk>" >> ${lang_dir}/lexicon.txt

    perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $lang_dir/lexicon.txt > $lang_dir/lexiconp.txt || exit 1
fi

if [ $stage -le 3 ]; then
  if ! grep "#0" $lang_dir/words.txt > /dev/null 2>&1; then
    max_word_id=$(tail -1 $lang_dir/words.txt | awk '{print $2}')
    echo "#0 $((max_word_id+1))" >> $lang_dir/words.txt
  fi
  ndisambig=$(utils/add_lex_disambig.pl --pron-probs $lang_dir/lexiconp.txt $lang_dir/lexiconp_disambig.txt)

  if ! grep "#0" $lang_dir/tokens.txt > /dev/null 2>&1 ; then
    max_token_id=$(tail -1 $lang_dir/tokens.txt | awk '{print $2}')
    for i in $(seq 0 $ndisambig); do
      echo "#$i $((i+max_token_id+1))"
    done >> $lang_dir/tokens.txt
  fi
fi
if [ $stage -le 4 ]; then
  if [ ! -f $lang_dir/L_disambig.fst.txt ]; then
    wdisambig_token=$(echo "#0" | utils/sym2int.pl $lang_dir/tokens.txt)
    wdisambig_word=$(echo "#0" | utils/sym2int.pl $lang_dir/words.txt)

    ${python} local/make_lexicon_fst.py \
      $lang_dir/lexiconp_disambig.txt | \
      utils/sym2int.pl --map-oov 1 -f 3 $lang_dir/tokens.txt | \
      utils/sym2int.pl -f 4 $lang_dir/words.txt  | \
      local/fstaddselfloops.pl $wdisambig_token $wdisambig_word > $lang_dir/L_disambig.fst.txt || exit 1
  fi

fi

if [ $stage -le 5 ]; then if [ ! -f $lang_dir/G.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/local/lm/lm_tgmed.arpa > $lang_dir/G.fst.txt
  else
    echo "Skip generating $lang_dir/G.fst.txt"
  fi
  if [ ! -f $lang_dir/G_4_gram.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      data/local/lm/lm_fglarge.arpa >$lang_dir/G_4_gram.fst.txt
  else
    echo "Skip generating data/lang_nosp/G_4_gram.fst.txt"
  fi
fi
