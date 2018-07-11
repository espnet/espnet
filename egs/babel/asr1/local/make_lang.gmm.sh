#!/bin/bash

. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands', 
set -e
set -u
set -o pipefail


dctdir=$1
tmpdir=$2
lang=$3


mkdir -p $lang

nlsyms=$dctdir/non_lang_syms.txt
dict=$dctdir/dct

# ------ Prepare files for FST composition
mkdir -p ${tmpdir}
rm -rf   ${tmpdir}/*

#the beginning of the lexicon is always the same
echo '!SIL sil' > ${tmpdir}/lexicon.txt
#the silence phone is the sil phone (more to come if in other datasets we have extra markings)
(   echo "sil"
    grep -v -e "<unk>" -e "<hes>" ${nlsyms} 
) > ${tmpdir}/silence_phones.txt
echo "sil"  > ${tmpdir}/optional_silence.txt

#cat $dict | sed 's:#:<hash>:g' >> ${tmpdir}/lexicon.txt
cat $dict | awk 'NF>1' |\
 grep -v "<\/s>" | grep -v "<s>" |\
awk '# Ensure constant formating acrros variants
{printf ("%s",$1); for (i=2;i<=NF;i++) printf " " $i; printf "\n"}' |\
 sort -u >> ${tmpdir}/lexicon.txt
    
./local/dict.PrintPhnSet.sh ${tmpdir}/lexicon.txt | grep -v -f ${tmpdir}/silence_phones.txt > ${tmpdir}/nonsilence_phones.txt
    
    
# --------------
echo "Preparing the language directory..."
#    --position-dependent-phones false \
tmpdir_lang=${tmpdir}.tmplang
echo "utils/prepare_lang.sh \
        ${tmpdir} \"<unk>\" ${tmpdir_lang} $lang"
utils/prepare_lang.sh \
    ${tmpdir} "<unk>" ${tmpdir_lang} $lang || exit 1



