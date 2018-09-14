. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

nlsyms=data/lang_1char/non_lang_syms.txt

#echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
#cat data/tr_*/text | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
#| sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
#wc -l ${dict}

mkdir -p dicts
langs="assamese tagalog swahili zulu bengali cantonese georgian haitian lao
pashto turkish kurmanji tokpisin vietnamese tamil"
for lang in $langs; do
    dict=dicts/${lang}_dict.txt
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/tr_babel_${lang}/text | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
done
