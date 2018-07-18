#!/bin/bash


# Prepare target babel languages 
# Assume that run.sh already run in $sourcedir_babel path

. ./path.sh
. ./cmd.sh
stage=0

data_in=data/train
data_fea=fbank/train
data_dmp=fbank/train/deltafalse

storage=/mnt/scratch01/tmp/karafiat/espnet/$RANDOM
# feature configuration
do_delta=false # true when using CNN
do_cvn=true

fbank_config=conf/fbank24.8k.conf
pitch_config=conf/pitch.8k.conf

echo "$0 $*"
. utils/parse_options.sh || exit 1;


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ${stage} -le 1 ]; then
    
    # Make Features if they are missing
    if [ ! -f $data_fea/feats.scp ]; then
        #[ ! -e $data_in/wav.scp.bak ] && sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" $data_in/wav.scp
	copy_data_dir.sh $data_in $data_fea; rm -f $data_fea/feats.scp ${data_fea}/cmvn.*
        steps/make_fbank_pitch.sh --cmd "$train_cmd --max_jobs_run 10" \
	    --fbank_config $fbank_config --pitch_config $pitch_config \
${data_fea} ${data_fea} ${data_fea}
        ./utils/fix_data_dir.sh ${data_fea} 
        compute-cmvn-stats scp:${data_fea}/feats.scp ${data_fea}/cmvn.ark
    fi
fi

if [ ${stage} -le 2 ]; then
    
    # Dump features on the $storage
    if [ ! -s  ${data_dmp}/feats.scp ]; then
	local/make_symlink_dir.sh --tmp-root $storage ${data_dmp}
	dump.sh --cmd "$train_cmd" --nj 20 --do_delta $do_delta --do_cvn $do_cvn \
	    ${data_fea}/feats.scp ${data_fea}/cmvn.ark ${data_dmp}/log ${data_dmp}
    fi
fi

