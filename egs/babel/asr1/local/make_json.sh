#!/bin/bash

. ./path.sh
. ./cmd.sh

data_in=
data=
lang=

. utils/parse_options.sh || exit 1;


nlsyms=$lang/non_lang_syms.txt
dict=$lang/train_units.txt

# make json labels
echo "data2json.sh --feat ${data_in}/feats.scp --nlsyms ${nlsyms} \
         ${data_in} ${dict} > ${data}/data.json" > ${data}/json.sge
chmod +x ${data}/json.sge
queue.pl --mem 20G ${data}/json.log ${data}/json.sge
