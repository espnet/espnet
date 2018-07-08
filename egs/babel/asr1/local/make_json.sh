#!/bin/bash

. ./path.sh
. ./cmd.sh

data_in=  # data/text is taken
data=     # data/feats.scp is taken
lang=     # $lang/non_lang_syms.txt and $lang/train_units.txt are taken

. utils/parse_options.sh || exit 1;


nlsyms=$lang/non_lang_syms.txt
dict=$lang/train_units.txt

# make json labels
echo "data2json.sh --feat ${data}/feats.scp --nlsyms ${nlsyms} \
         ${data_in} ${dict} > ${data}/data.json" > ${data}/json.sge
chmod +x ${data}/json.sge
queue.pl --mem 20G ${data}/json.log ${data}/json.sge
