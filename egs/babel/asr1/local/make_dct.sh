#!/bin/bash

. ./path.sh
. ./cmd.sh


data=
lang=

. utils/parse_options.sh || exit 1;


mkdir -p $lang

nlsyms=$lang/non_lang_syms.txt
dict=$lang/train_units.txt

# For language independent character set we also include target data
echo "make a non-linguistic symbol list"
for d in $data; do
    cat $d/text 
done | cut -f 2- | \
    tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
#    grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
cat ${nlsyms}

echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

for d in $data; do
    cat $d/text
done | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | grep -v '<unk>' | awk '{print $0 " " NR+1}' >> ${dict}
wc -l ${dict}
#| text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
    #| sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}

