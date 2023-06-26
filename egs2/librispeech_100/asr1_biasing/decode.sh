. ./path.sh
. ./cmd.sh
export PYTHONPATH="/home/gs534/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/opensource/espnet:$PYTHONPATH"
echo $PYTHONPATH

JOB=1
#$1
nj=128

inference_config=conf/decode_asr.yaml
dset="test_other"
asr_exp=exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_suffix
inference_tag=decode_b20_nolm_avebest
_dir=${asr_exp}/${inference_tag}/${dset}
_logdir="${_dir}/logdir"
mkdir -p $_logdir

split_scps=""
for n in $(seq "${nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
# Split nj scps
echo ${split_scps}
utils/split_scp.pl "dump/raw/${dset}/wav.scp" ${split_scps}

inference_asr_model=valid.loss.ave_10best.pth
# inference_asr_model=latest.pth
perutt_blist=data/${dset}/perutt_blist.json


python -m espnet2.bin.asr_inference \
    --batch_size 1 \
    --ngpu 0 \
    --data_path_and_name_and_type dump/raw/${dset}/wav.scp,speech,kaldi_ark \
    --key_file "${_logdir}"/keys.$JOB.scp \
    --asr_train_config "${asr_exp}"/config.yaml \
    --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
    --output_dir "${_logdir}"/output.$JOB \
    --config conf/decode_asr.yaml \
    --perutt_blist $perutt_blist \
    --biasinglist data/Blist/all_rare_words.txt \
    --bmaxlen 1000 \
