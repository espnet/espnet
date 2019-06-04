#!/bin/bash

# Copyright 2019 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

results=""
# e.g., "exp/tr_it_pytorch_train/decode_dt_it_decode/result.wrd.txt
#        exp/tr_it_pytorch_train/decode_et_it_decode/result.wrd.txt"'
lm=""
outdir=""

. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: $0 <tr_conf> <rec_conf> <cmvn> <e2e>, for example:";
    echo "<tr_conf>:  conf/train.yaml"
    echo "<rec_conf>: conf/decode.yaml"
    echo "<cmvn>:     data/tr_it/cmvn.ark"
    echo "<e2e>:      exp/tr_it_pytorch_train/results/model.last10.avg.best"
    exit 1;
fi

tr_conf=$1
rec_conf=$2
cmvn=$3
e2e=$4

if [ -z ${outdir} ]; then
    dir=archive
else
    dir=${outdir}
fi
#echo "will write files in ${PWD}/${dir}"
mkdir -p ${dir}/{cmvn,conf,lm,e2e,results}

echo "  - Model files (archived to ${dir}.tgz by \`\$ pack_model.sh\`)"
echo "    - model link: (put the model link manually. please contact Shinji Watanabe <shinjiw@ieee.org> if you want a web storage to put your files)"

# configs
if [ -e ${tr_conf} ]; then
    cp -L ${tr_conf} ${dir}/conf/
    echo -n "    - training config file: \`"
    echo ${tr_conf} | sed -e "s/$/\`/" 
else
    echo "missing ${tr_conf}"
    exit 1
fi
if [ -e ${rec_conf} ]; then
    cp -L ${rec_conf} ${dir}/conf/
    echo -n "    - decoding config file: \`"
    echo ${rec_conf} | sed -e "s/$/\`/" 
else
    echo "missing ${rec_conf}"
    exit 1
fi

# cmvn
if [ -e ${cmvn} ]; then
    cp -L ${cmvn} ${dir}/cmvn/
    echo -n "    - cmvn file: \`"
    echo ${cmvn} | sed -e "s/$/\`/" 
else
    echo "missing ${cmvn}"
    exit 1
fi

# e2e
if [ -e ${e2e} ]; then
    cp -L ${e2e} ${dir}/e2e/
    echo -n "    - e2e file: \`"
    echo ${e2e} | sed -e "s/$/\`/"

    e2e_conf=$(dirname ${e2e})/model.json
    if [ ! -e ${e2e_conf} ]; then
	echo missing ${e2e_conf}
	exit 1
    else
	echo -n "    - e2e JSON file: \`"
	echo ${e2e_conf} | sed -e "s/$/\`/"
	cp ${e2e_conf} ${dir}/e2e/
    fi
else
    echo "missing ${e2e}"
    exit 1
fi

# lm
if [ -n "${lm}" ]; then
    if [ -e ${lm} ]; then
	cp -L ${lm} ${dir}/lm/
	echo -n "    - lm file: \`"
	echo ${lm} | sed -e "s/$/\`/"

	lm_conf=`dirname ${lm}`/model.json
	if [ ! -e ${lm_conf} ]; then
	    echo missing ${lm_conf}
	    exit 1
	else
	    echo -n "    - lm JSON file: \`"
	    echo ${lm_conf} | sed -e "s/$/\`/"
	    cp ${lm_conf} ${dir}/lm/
	fi
    else
	echo "missing ${lm}"
	exit 1
    fi
fi

# compress
# echo "compress model files to ${dir}.tgz"
tar zcvf ${dir}.tgz ${dir} > /dev/null

# results
if [ -n "${results}" ]; then
    echo "  - Results (paste them by yourself or obtained by \`\$ pack_model.sh --results <results>\`)"
fi
for x in ${results}; do
    if [ -e ${x} ]; then
	echo "\`\`\`"
	echo "${x}"
	grep -e Avg -e SPKR -m 2 ${x}
	echo "\`\`\`"
    else
	echo "missing ${x}"
	exit 1
    fi
done

exit 0
