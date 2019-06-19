#!/bin/bash

# Copyright 2019 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

results=""
# e.g., "exp/tr_it_pytorch_train/decode_dt_it_decode/result.wrd.txt
#        exp/tr_it_pytorch_train/decode_et_it_decode/result.wrd.txt"'
lm=""
dict=""
etc=""
outfile="model"

. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <tr_conf> <dec_conf> <cmvn> <e2e>, for example:"
    echo "<tr_conf>:  conf/train.yaml"
    echo "<dec_conf>: conf/decode.yaml"
    echo "<cmvn>:     data/tr_it/cmvn.ark"
    echo "<e2e>:      exp/tr_it_pytorch_train/results/model.last10.avg.best"
    exit 1
fi

tr_conf=$1
dec_conf=$2
cmvn=$3
e2e=$4

echo "  - Model files (archived to ${outfile}.tar.gz by \`\$ pack_model.sh\`)"
echo "    - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)"

# configs
if [ -e ${tr_conf} ]; then
    tar cfh ${outfile}.tar ${tr_conf}
    echo -n "    - training config file: \`"
    echo ${tr_conf} | sed -e "s/$/\`/"
else
    echo "missing ${tr_conf}"
    exit 1
fi
if [ -e ${dec_conf} ]; then
    tar rfh ${outfile}.tar ${dec_conf}
    echo -n "    - decoding config file: \`"
    echo ${dec_conf} | sed -e "s/$/\`/"
else
    echo "missing ${dec_conf}"
    exit 1
fi

# cmvn
if [ -e ${cmvn} ]; then
    tar rfh ${outfile}.tar ${cmvn}
    echo -n "    - cmvn file: \`"
    echo ${cmvn} | sed -e "s/$/\`/"
else
    echo "missing ${cmvn}"
    exit 1
fi

# e2e
if [ -e ${e2e} ]; then
    tar rfh ${outfile}.tar ${e2e}
    echo -n "    - e2e file: \`"
    echo ${e2e} | sed -e "s/$/\`/"

    e2e_conf=$(dirname ${e2e})/model.json
    if [ ! -e ${e2e_conf} ]; then
	echo missing ${e2e_conf}
	exit 1
    else
	echo -n "    - e2e JSON file: \`"
	echo ${e2e_conf} | sed -e "s/$/\`/"
	tar rfh ${outfile}.tar ${e2e_conf}
    fi
else
    echo "missing ${e2e}"
    exit 1
fi

# lm
if [ -n "${lm}" ]; then
    if [ -e ${lm} ]; then
	tar rfh ${outfile}.tar ${lm}
	echo -n "    - lm file: \`"
	echo ${lm} | sed -e "s/$/\`/"

	lm_conf=$(dirname ${lm})/model.json
	if [ ! -e ${lm_conf} ]; then
	    echo missing ${lm_conf}
	    exit 1
	else
	    echo -n "    - lm JSON file: \`"
	    echo ${lm_conf} | sed -e "s/$/\`/"
	    tar rfh ${outfile}.tar ${lm_conf}
	fi
    else
	echo "missing ${lm}"
	exit 1
    fi
fi

# dict
if [ -n "${dict}" ]; then
    if [ -e ${dict} ]; then
	tar rfh ${outfile}.tar ${dict}
	echo -n "    - dict file: \`"
	echo ${dict} | sed -e "s/$/\`/"
    else
	echo "missing ${dict}"
	exit 1
    fi
fi

# etc
for x in ${etc}; do
    if [ -e ${x} ]; then
	tar rfh ${outfile}.tar ${x}
	echo -n "    - etc file: \`"
	echo ${x} | sed -e "s/$/\`/"
    else
	echo "missing ${x}"
	exit 1
    fi
done

# finally compress the tar file
gzip -f ${outfile}.tar

# results
if [ -n "${results}" ]; then
    echo "  - Results (paste them by yourself or obtained by \`\$ pack_model.sh --results <results>\`)"
    echo "\`\`\`"
fi
for x in ${results}; do
    if [ -e ${x} ]; then
	echo "${x}"
	grep -e Avg -e SPKR -m 2 ${x}
    else
	echo "missing ${x}"
	exit 1
    fi
done
if [ -n "${results}" ]; then
    echo "\`\`\`"
fi

exit 0
