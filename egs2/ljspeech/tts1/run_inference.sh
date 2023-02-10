if [ -z "$1" ]; then
    pth_file="valid.loss.best.pth"
else
    pth_file="$1"
fi

if [ -z "$2" ]; then
    dataset="eval1"
else
    dataset="$2"
fi


for x in exp/fastspeech2_0 exp/*max75*; do
    if [ "${x}" != "exp/test" ] && [ -e "${x}/${pth_file}" ]; then
        decode_dir="${x}/decode_fastspeech_${pth_file:0:-4}"
        feats_dir="${decode_dir}/${dataset}"
        out_dir="${feats_dir}_parallel_wavegan"
        if [ -e "${feats_dir}" ]; then
            echo "${x} has been decoded, skipping inference"
        else
            echo "Running inference for ${x}"
            ./run_mfa.sh --stage 7 --stop-stage 7 \
                --tts_exp "${x}" \
                --inference_config conf/tuning/decode_fastspeech.yaml \
                --inference_model "${pth_file}" \
                --inference_nj 1 \
                --test_sets "${dataset}" \
                --gpu_inference true
        fi
        if [ -e "${out_dir}" ]; then
            echo "${out_dir} exists, skipping generation"
        else
            echo "Generating wavs for ${x}"
            parallel-wavegan-decode \
                --checkpoint parallel_wavegan/checkpoint-3000000steps.pkl \
                --feats-scp "${feats_dir}/norm/feats.scp" \
                --outdir "${out_dir}"
        fi
    fi
done



