#!/bin/bash

. ./path.sh
. ./cmd.sh


add_langinfo=""

. utils/parse_options.sh || exit 1;
data=$1
dctdir=$2

mkdir -p $dctdir

nlsyms=$dctdir/non_lang_syms.txt
dict=$dctdir/dct

# For language independent character set we also include target data
echo "make a non-linguistic symbol list"
for d in $data; do
    cat $d/text 
done | awk '{for(i=2;i<=NF;i++) W[$i]=1}; END{for( w in W) print w}' > $dctdir/text.wlist
grep -e '^<.*>$'  $dctdir/text.wlist > ${nlsyms} 

# |cut 
# |cut -f 2- | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
# |cut -f 2- | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
cat ${nlsyms}

echo "make a pron variants"
cat $dctdir/text.wlist |\
 text2token.py -s 0 -n 1 -l ${nlsyms} |\
 awk -v add_langinfo="$add_langinfo" '
k==0                      {W[$1]=1}  
k==1 &&  add_langinfo!="" {for(i=1;i<=NF;i++) if(!($i in W )) $i = $i add_langinfo}
k==1                      {print}' k=0 ${nlsyms} k=1 /dev/stdin  > $dctdir/text.wlist.pron

echo "make the dict"
paste $dctdir/text.wlist $dctdir/text.wlist.pron > ${dict}
wc -l ${dict}

