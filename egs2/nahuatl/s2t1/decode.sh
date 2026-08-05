#!/usr/bin/env bash
# Decode + score the three single-region test sets, then combine into an
# aggregate score.
#
# Why per region: espnet2's s2t_inference applies ONE --lang_sym to every
# utterance in a decode run, but the correct region tag differs per recording.
# So we decode each single-region test set with its own lang_sym. The mixed
# "nahuatl_test" set cannot be decoded correctly with a single lang_sym; its
# score is exactly the union of the three regions, which we reconstruct here by
# concatenating the per-region hyp/ref and re-scoring.
#SBATCH -N 1 -n 1 -p gpuA40x4,gpuA100x4
#SBATCH --gres=gpu:1 -c 16 --mem 60000M
#SBATCH --account=bbjs-delta-gpu
#SBATCH --time=4:00:00
#SBATCH --job-name=nahuatl-decode
#SBATCH --output=%x_%j.log
set -o pipefail

RECIPE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$RECIPE_DIR"
source path.sh

inference_config=conf/decode_owsm.yaml

# region -> lang_sym
regions=(hidalgo orizaba_zongolica zacatlan_tepetzintla)
declare -A LANG=(
    [hidalgo]="<nah_hid>"
    [orizaba_zongolica]="<nah_ozg>"
    [zacatlan_tepetzintla]="<nah_ztp>"
)

for region in "${regions[@]}"; do
    echo "=== Decoding nahuatl_${region}_test with lang_sym=${LANG[$region]} ==="
    bash run.sh --stage 12 --stop_stage 13 \
        --test_sets "nahuatl_${region}_test" \
        --inference_config "${inference_config}" \
        --inference_args "--lang_sym ${LANG[$region]}"
done

# ── Combine the three regions into an aggregate CER (== the mixed test set) ───
# Locate the score_cer dirs produced above (one per region) and concatenate.
s2t_exp=$(ls -d exp/s2t_train_owsm_v4_nahuatl_raw_bpe50000_init_param* 2>/dev/null | head -1)
for base in "${s2t_exp}"/*/nahuatl_hidalgo_test/score_cer; do
    [ -d "$base" ] || continue
    inf_dir=$(dirname "$(dirname "$base")")
    out="${inf_dir}/nahuatl_combined_score_cer"
    mkdir -p "$out"
    for f in hyp ref; do
        cat "${inf_dir}"/nahuatl_hidalgo_test/score_cer/${f}.trn \
            "${inf_dir}"/nahuatl_orizaba_zongolica_test/score_cer/${f}.trn \
            "${inf_dir}"/nahuatl_zacatlan_tepetzintla_test/score_cer/${f}.trn \
            > "${out}/${f}.trn"
    done
    sclite -r "${out}/ref.trn" trn -h "${out}/hyp.trn" trn -i rm -o all stdout \
        > "${out}/result.txt"
    echo "=== Combined (all 3 regions) CER ==="
    grep -e Avg -e SPKR -m 2 "${out}/result.txt"
    echo "Full result: ${out}/result.txt"
done
