srctexts="data/${train_set}/text"
test_set="eval1"
data_feats="dump/raw"
_data=${data_feats}/${test_set}
tts_exp=exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space

inference_nj=16
nj=32
gpu_inference=true

. ./path.sh
. ./cmd.sh

# VERSA eval related
skip_scoring=true # Skip scoring stages.
skip_wer=true # Skip WER evaluation.
whisper_tag=medium # Whisper model tag.
whisper_dir=local/whisper # Whisper model directory.
cleaner=whisper_en # Text cleaner for whisper model.
hyp_cleaner=whisper_en # Text cleaner for hypothesis.

versa_config=conf/versa.yaml # VERSA evaluation configuration.

inference_tag=decode_vits_latest

_gen_dir=${tts_exp}/${inference_tag}/${test_set}

if ! ${skip_wer}; then
    ./scripts/utils/evaluate_asr.sh \
        --whisper_tag ${whisper_tag} \
        --whisper_dir ${whisper_dir} \
        --cleaner ${cleaner} \
        --hyp_cleaner ${hyp_cleaner} \
        --inference_nj ${inference_nj} \
        --nj ${nj} \
        --gt_text ${_data}/text \
        --gpu_inference ${gpu_inference} \
        ${_gen_dir}/wav/wav_test.scp ${_gen_dir}/scoring/eval_wer
fi

_opts=
_eval_dir=${_gen_dir}/scoring/versa_eval
mkdir -p ${_eval_dir}

_pred_file=${_gen_dir}/wav/wav_test.scp
_score_config=${versa_config}
_gt_file=${_data}/wav_test.scp

_nj=$(( ${inference_nj} < $(wc -l < ${_pred_file}) ? ${inference_nj} : $(wc -l < ${_pred_file}) ))

_split_files=""
for n in $(seq ${_nj}); do
    _split_files+="${_eval_dir}/pred.${n} "
done
utils/split_scp.pl ${_pred_file} ${_split_files}

if [ -n "${_gt_file}" ]; then
    _split_files=""
    for n in $(seq ${_nj}); do
        _split_files+="${_eval_dir}/gt.${n} "
    done
    utils/split_scp.pl ${_gt_file} ${_split_files}
    _opts+="--gt ${_eval_dir}/gt.JOB"
fi

if ${gpu_inference}; then
    _cmd="${cuda_cmd}"
    _ngpu=1
else
    _cmd="${decode_cmd}"
    _ngpu=0
fi

${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_eval_dir}"/versa_eval.JOB.log \
python -m versa.bin.scorer \
    --pred ${_eval_dir}/pred.JOB \
    --score_config ${_score_config} \
    --cache_folder ${_eval_dir}/cache \
    --use_gpu ${gpu_inference} \
    --output_file ${_eval_dir}/result.JOB.txt \
    --io soundfile \
    ${_opts} 2>&1 | grep -i "info" || exit 1;

python pyscripts/utils/aggregate_tts_eval.py \
    --logdir ${_eval_dir} \
    --scoredir ${_eval_dir} \
    --nj ${_nj}