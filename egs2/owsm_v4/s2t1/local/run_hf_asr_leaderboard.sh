#!/usr/bin/env bash
# Evaluate an OWSM checkpoint on the Open ASR Leaderboard test sets.
# Supports full leaderboard evaluation (all utterances across 9 benchmark configs)
# or a quick 6-set, 100-utterance sample smoke test for local verification.
#
# The dataset is gated: accept the terms at
# https://huggingface.co/datasets/hf-audio/open-asr-leaderboard and log in
# with `huggingface-cli login` (or set HF_TOKEN) before running.

set -euo pipefail

help_message=$(cat << 'EOF'
Usage: local/run_hf_asr_leaderboard.sh [options]

Evaluate an OWSM checkpoint on Open ASR Leaderboard test sets.

Options:
  --subset (full|small)       Evaluation profile (default: full)
                              full: 9 benchmark sets, all utterances (n=0)
                              small: 6 benchmark sets, 100 utterances each
  --model_tag <tag>           Hugging Face model tag (default: espnet/owsm_v4_base_102M)
  --n <n>                     Utterances per set (default: 0 for full, 100 for small)
  --seed <seed>               Random seed for sampling (default: 0)
  --batch_size <int>          Minibatch size for decoding (default: 8)
  --beam_size <int>           Beam size (default: 5)
  --maxlenratio <float>       Max length ratio (default: 0.0)
  --sort (true|false)         Sort utterances by length before batching (default: true)
  --threads <int>             PyTorch intra-op threads (default: 4)
  --quantize (true|false)     Dynamic int8 quantization (default: false)
  --device (cpu|cuda)         Inference device (default: cpu)
  --dtype (float32|float16)   PyTorch dtype (default: float32)
  --sets "<set1> <set2>..."   Custom list of "config:split" pairs
  --outdir <dir>              Output directory (default: exp/hf_asr_leaderboard)
EOF
)

subset=full  # 'full' (all 9 English leaderboard sets, n=0) or 'small' (6 sets, n=100 sample)
model_tag=espnet/owsm_v4_base_102M
n=
seed=0
batch_size=8
beam_size=5
maxlenratio=0.0
sort=true
threads=4
quantize=false
device=cpu
dtype=float32
max_dur=
outdir=exp/hf_asr_leaderboard
sets=
revision=b6bdcd0beb34f8975dc659796176d88f43aff502  # pinned dataset commit

. utils/parse_options.sh
. ./path.sh

if [ "${subset}" = "small" ]; then
    [ -z "${n}" ] && n=100
    [ -z "${max_dur}" ] && max_dur=30.0
    [ -z "${sets}" ] && sets="librispeech:test.clean librispeech:test.other ami_cleaned:test voxpopuli_cleaned_aa:test earnings22:test common_voice:test"
elif [ "${subset}" = "full" ]; then
    [ -z "${n}" ] && n=0
    [ -z "${max_dur}" ] && max_dur=0.0
    [ -z "${sets}" ] && sets="librispeech:test.clean librispeech:test.other ami_cleaned:test voxpopuli_cleaned_aa:test common_voice:test tedlium:test gigaspeech_cleaned:test spgispeech:test earnings22:test"
else
    [ -z "${n}" ] && n=0
    [ -z "${max_dur}" ] && max_dur=0.0
fi

mkdir -p "${outdir}"
summary="${outdir}/summary.md"
{
    if [ "${n}" -gt 0 ]; then
        echo "Model: ${model_tag}, n=${n} per set (seed=${seed}), subset=${subset}, device=${device}, threads=${threads}"
    else
        echo "Model: ${model_tag}, full evaluation (all utterances), subset=${subset}, device=${device}, threads=${threads}"
    fi
    echo
    echo "| set | model | decoding | WER (%) | RTFx | s/utt |"
    echo "| --- | --- | --- | ---: | ---: | ---: |"
} > "${summary}"

for pair in ${sets}; do
    dataset=${pair%%:*}
    split=${pair#*:}
    name=$(echo "lb_${dataset}_${split}" | tr -c 'A-Za-z0-9_\n' '_')
    data="data/${name}"
    # reuse a prepared set only if it is complete and was drawn with the matching parameters
    if [ ! -f "${data}/wav.scp" ] || [ ! -f "${data}/text" ] || ! python3 - "${data}/info.json" "${n}" "${seed}" "${revision}" "${max_dur}" <<'EOF'
import json, sys
try:
    info = json.load(open(sys.argv[1]))
    req_n = int(sys.argv[2])
    req_seed = int(sys.argv[3])
    req_rev = sys.argv[4]
    req_max_dur = float(sys.argv[5])

    if req_n > 0:
        ok = (info.get("n") == req_n and
              info.get("seed") == req_seed and
              info.get("revision") == req_rev and
              abs(info.get("max_dur", 30.0) - req_max_dur) < 1e-3)
    else:
        ok = (info.get("n_requested", 0) == 0 and
              info.get("revision") == req_rev and
              info.get("max_dur", 0.0) <= 0)
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
EOF
    then
        echo "=== preparing ${data}"
        rm -rf "${data}"
        python3 local/prepare_hf_asr_leaderboard.py \
            --dataset "${dataset}" --split "${split}" --n "${n}" --seed "${seed}" \
            --max_dur "${max_dur}" --revision "${revision}" --out "${data}"
    fi
    echo "=== decoding ${data}"
    python3 local/eval_hf_asr_leaderboard.py \
        --data "${data}" --model_tag "${model_tag}" --device "${device}" \
        --dtype "${dtype}" --batch_size "${batch_size}" --beam_size "${beam_size}" \
        --maxlenratio "${maxlenratio}" --sort "${sort}" --threads "${threads}" \
        --quantize "${quantize}" \
        --out "${outdir}/${name}__${model_tag##*/}__bs${batch_size}_beam${beam_size}.json" \
        | tail -n 1 >> "${summary}"
done

echo
cat "${summary}"
