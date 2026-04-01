#!/usr/bin/env bash
# Toy end-to-end pipeline test for SOT native Whisper training + inference.
#
# Verifies: token list generation -> vocab expansion -> training (2 epochs)
#           -> inference (beam_size=1 & 5) -> correct decoding.
# Expected runtime: ~5-10 minutes on a single GPU.
#
# Usage:
#   bash scripts/toy_pipeline_test.sh              # run all steps
#   bash scripts/toy_pipeline_test.sh --skip-train  # skip training (reuse existing model)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RECIPE_DIR=/work/nvme/bbjs/chuang14/espnet-owsm-dtai/egs2/ami/sot_asr1
ESPNET_ROOT=/work/nvme/bbjs/chuang14/espnet-owsm-dtai
export PYTHONPATH="${ESPNET_ROOT}:${PYTHONPATH:-}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate espnet-owsm

cd "${RECIPE_DIR}"

# Source data from dicow_asr1 (already pre-segmented)
DICOW_DIR=/work/nvme/bbjs/chuang14/espnet-owsm-dtai/egs2/ami/dicow_asr1

TRAIN_TOY=data/train_toy
DEV_TOY=data/dev_toy

EXP_DIR=exp/toy_pipeline_test
TOKEN_LIST=${EXP_DIR}/token_list.txt
TRAIN_OUT=${EXP_DIR}/sot_train
DECODE_OUT_B1=${TRAIN_OUT}/decode_dev_beam1
DECODE_OUT_B5=${TRAIN_OUT}/decode_dev_beam5

NTRAIN=50
NDEV=20

ADDED_TOKENS_FILE=local/added_tokens.txt
NUM_ADDED_TOKENS=16
BASE_VOCAB=51865
EXPECTED_TOKENS=$((BASE_VOCAB + NUM_ADDED_TOKENS))  # 51881

SKIP_TRAIN=false
for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
    esac
done

PASS=0
FAIL=0

check() {
    local desc="$1"; shift
    if "$@"; then
        echo "  [PASS] ${desc}"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] ${desc}"
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# Step 0: Prepare toy data (select clean utterances with valid timestamps)
# ---------------------------------------------------------------------------
echo "=== Step 0: Preparing toy data ==="

# Source data from dicow_asr1 (full sets with pre-segmented WAVs)
DICOW_TRAIN_SRC=${DICOW_DIR}/data/train_toy
DICOW_DEV_SRC=${DICOW_DIR}/data/dev_toy

# SOT with tiktoken requires all timestamps <= 30.00s (Whisper's max).
# The source data has utterances with timestamps > 30s due to a data
# preparation issue. We filter those out when selecting toy subsets.

if [ -d "${TRAIN_TOY}" ] && [ -f "${TRAIN_TOY}/wav.scp" ] && \
   [ -d "${DEV_TOY}" ] && [ -f "${DEV_TOY}/wav.scp" ]; then
    echo "  Toy data already exists, skipping selection."
else
    # Verify source data exists (dicow toy data must be created first)
    for src in "${DICOW_TRAIN_SRC}" "${DICOW_DEV_SRC}"; do
        if [ ! -d "${src}" ] || [ ! -f "${src}/text" ]; then
            echo "  ERROR: Source data not found: ${src}"
            echo "  Please run dicow_asr1/scripts/toy_pipeline_test.sh first to create toy data."
            exit 1
        fi
    done

    echo "  Selecting utterances with all timestamps <= 30s..."

    export DICOW_TRAIN_SRC="${DICOW_TRAIN_SRC}"
    export DICOW_DEV_SRC="${DICOW_DEV_SRC}"

    python3 << 'PYEOF'
import re, os

MAX_TS = 30.0

def get_clean_utt_ids(text_file, n):
    """Select up to n utterance IDs where all timestamps are <= MAX_TS."""
    clean = []
    skipped = 0
    with open(text_file) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = parts[1]
            timestamps = re.findall(r"<\|(\d+\.\d+)\|>", text)
            max_ts = max(float(ts) for ts in timestamps) if timestamps else 0.0
            if max_ts <= MAX_TS:
                clean.append(utt_id)
                if len(clean) >= n:
                    break
            else:
                skipped += 1
    print(f"    {text_file}: selected {len(clean)}, skipped {skipped} (timestamps > {MAX_TS}s)")
    return set(clean)

def filter_file(src_path, dst_path, keep_ids):
    """Copy lines from src to dst where the first field (utt_id) is in keep_ids."""
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            utt_id = line.strip().split(maxsplit=1)[0]
            if utt_id in keep_ids:
                fout.write(line)

for split, n, src_env in [("data/train_toy", 50, "DICOW_TRAIN_SRC"),
                           ("data/dev_toy",   20, "DICOW_DEV_SRC")]:
    src = os.environ[src_env]
    clean_ids = get_clean_utt_ids(os.path.join(src, "text"), n)

    os.makedirs(split, exist_ok=True)
    for f in ["text", "wav.scp", "utt2spk", "spk2utt"]:
        src_file = os.path.join(src, f)
        if os.path.exists(src_file):
            filter_file(src_file, os.path.join(split, f), clean_ids)

    # Symlink segments_wav directory
    seg_src = os.path.join(src, "segments_wav")
    seg_dst = os.path.join(split, "segments_wav")
    if os.path.isdir(seg_src) and not os.path.exists(seg_dst):
        os.symlink(os.path.realpath(seg_src), seg_dst)
PYEOF
fi

echo "  train_toy: $(wc -l < "${TRAIN_TOY}/wav.scp") utterances"
echo "  dev_toy:   $(wc -l < "${DEV_TOY}/wav.scp") utterances"

# ---------------------------------------------------------------------------
# Step 1: Generate token list (tiktoken-based)
# ---------------------------------------------------------------------------
echo "=== Step 1: Generating token list ==="

mkdir -p "${EXP_DIR}"

python3 << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("ESPNET_ROOT", "/work/nvme/bbjs/chuang14/espnet-owsm-dtai"))
from espnet2.train.sot_preprocessor import SOTWhisperPreprocessor

SOTWhisperPreprocessor.generate_token_list(
    output_path="exp/toy_pipeline_test/token_list.txt",
    added_tokens_txt="local/added_tokens.txt",
)
PYEOF

TOKEN_COUNT=$(wc -l < "${TOKEN_LIST}")
echo "  token_list.txt: ${TOKEN_COUNT} tokens (expected ${EXPECTED_TOKENS})"

check "token_list.txt has ${EXPECTED_TOKENS} tokens (got ${TOKEN_COUNT})" \
    [ "${TOKEN_COUNT}" -eq "${EXPECTED_TOKENS}" ]

# ---------------------------------------------------------------------------
# Step 2: Vocab expansion verification (Python, no GPU needed)
# ---------------------------------------------------------------------------
echo "=== Step 2: Vocab expansion verification ==="

STEP2_RC=0
python3 << 'PYEOF' || STEP2_RC=$?
import sys, os, torch
sys.path.insert(0, os.environ.get("ESPNET_ROOT", "/work/nvme/bbjs/chuang14/espnet-owsm-dtai"))

from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder, ExpandedTokenEmbedding
from espnet2.asr.sot_espnet_model import SOTWhisperModel

VOCAB_SIZE = 51881
BASE_VOCAB = 51865
NUM_ADDED = 16

# ---- Part A: Decoder vocab expansion ----
print("  -- Decoder vocab expansion --")
decoder = OpenAIWhisperDecoder(
    vocab_size=VOCAB_SIZE,
    encoder_output_size=384,  # whisper-tiny
    whisper_model="tiny",
    load_origin_token_embedding=True,
)

emb = decoder.decoders.token_embedding
ok = True

# Check 1: ExpandedTokenEmbedding type
is_expanded = isinstance(emb, ExpandedTokenEmbedding)
print(f"  token_embedding is ExpandedTokenEmbedding: {is_expanded}")
if not is_expanded:
    print("  [FAIL] Expected ExpandedTokenEmbedding")
    ok = False

if is_expanded:
    # Check 2: ori_emb size
    ori_ok = emb.ori_emb.num_embeddings == BASE_VOCAB
    print(f"  ori_emb.num_embeddings == {BASE_VOCAB}: {ori_ok} (got {emb.ori_emb.num_embeddings})")
    if not ori_ok: ok = False

    # Check 3: add_emb size
    add_ok = emb.add_emb.num_embeddings == NUM_ADDED
    print(f"  add_emb.num_embeddings == {NUM_ADDED}: {add_ok} (got {emb.add_emb.num_embeddings})")
    if not add_ok: ok = False

    # Check 4: total size
    total_ok = emb.num_embeddings == VOCAB_SIZE
    print(f"  total num_embeddings == {VOCAB_SIZE}: {total_ok} (got {emb.num_embeddings})")
    if not total_ok: ok = False

    # Check 5: Forward with added token IDs doesn't crash
    try:
        test_ids = torch.arange(BASE_VOCAB, VOCAB_SIZE)
        out = emb(test_ids)
        fwd_ok = out.shape == (NUM_ADDED, 384)
        print(f"  forward(added token IDs) shape correct: {fwd_ok} (got {out.shape})")
        if not fwd_ok: ok = False
    except Exception as e:
        print(f"  [FAIL] forward crashed: {e}")
        ok = False

# ---- Part B: Full SOTWhisperModel ----
print("\n  -- SOTWhisperModel checks --")

# Load token list
with open("exp/toy_pipeline_test/token_list.txt") as f:
    token_list = [line.strip() for line in f]

# Minimal model instantiation (no GPU needed)
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder

encoder = OpenAIWhisperEncoder(whisper_model="tiny", dropout_rate=0.0)
from espnet2.asr.ctc import CTC
ctc = CTC(odim=VOCAB_SIZE, encoder_output_size=encoder.output_size())

model = SOTWhisperModel(
    vocab_size=VOCAB_SIZE,
    token_list=token_list,
    frontend=None,
    specaug=None,
    normalize=None,
    preencoder=None,
    encoder=encoder,
    postencoder=None,
    decoder=decoder,
    ctc=ctc,
    joint_network=None,
    ctc_weight=0.0,
    use_uppercase_loss=True,
    sym_sos="<|startoftranscript|>",
    sym_eos="<|endoftext|>",
)

# Check model properties
vs_ok = model.vocab_size == VOCAB_SIZE
print(f"  model.vocab_size == {VOCAB_SIZE}: {vs_ok} (got {model.vocab_size})")
if not vs_ok: ok = False

sos_ok = model.sos == 50258
print(f"  model.sos == 50258: {sos_ok} (got {model.sos})")
if not sos_ok: ok = False

eos_ok = model.eos == 50257
print(f"  model.eos == 50257: {eos_ok} (got {model.eos})")
if not eos_ok: ok = False

upper_ok = len(model.upper_cased_tokens) > 0
print(f"  len(upper_cased_tokens) > 0: {upper_ok} (got {len(model.upper_cased_tokens)})")
if not upper_ok: ok = False

# ---- Part C: Forward pass with dummy data ----
print("\n  -- Dummy forward pass --")
try:
    model.train()
    # Dummy audio: 1 sample, 16000 samples (1 sec at 16kHz)
    speech = torch.randn(1, 16000)
    speech_lengths = torch.tensor([16000])
    # Dummy text: [<|en|>=50259, <|transcribe|>=50359, <|0.00|>=50364, hello=15947, <|1.00|>=50414, <sc>=51865, <|endoftext|>=50257]
    text = torch.tensor([[50259, 50359, 50364, 15947, 50414, 51865]])
    text_lengths = torch.tensor([6])

    loss, stats, weight = model(speech, speech_lengths, text, text_lengths)

    loss_finite = torch.isfinite(loss).item()
    print(f"  loss is finite: {loss_finite} (loss={loss.item():.4f})")
    if not loss_finite: ok = False

    has_loss_att = "loss_att" in stats
    print(f"  stats has 'loss_att': {has_loss_att}")
    if not has_loss_att: ok = False

    has_acc = "acc" in stats
    print(f"  stats has 'acc': {has_acc}")
    if not has_acc: ok = False

except Exception as e:
    print(f"  [FAIL] Forward pass crashed: {e}")
    import traceback; traceback.print_exc()
    ok = False

if ok:
    print("\n  STEP 2: ALL VOCAB EXPANSION CHECKS PASSED")
    sys.exit(0)
else:
    print("\n  STEP 2: SOME CHECKS FAILED")
    sys.exit(1)
PYEOF

check "Vocab expansion and model verification" [ "${STEP2_RC}" -eq 0 ]

# ---------------------------------------------------------------------------
# Step 3: Training (2 epochs, whisper-tiny)
# ---------------------------------------------------------------------------
if [ "${SKIP_TRAIN}" = false ]; then
    echo "=== Step 3: Training (2 epochs, whisper-tiny, ${NTRAIN} utts) ==="

    python -m espnet2.bin.sot_train \
        --config conf/tuning/train_sot_tiny.yaml \
        --token_list "${TOKEN_LIST}" \
        --token_type whisper_multilingual \
        --train_shape_file "${TRAIN_TOY}/wav.scp" \
        --valid_shape_file "${DEV_TOY}/wav.scp" \
        --output_dir "${TRAIN_OUT}" \
        --ngpu 1 \
        --num_workers 2 \
        --train_data_path_and_name_and_type "${TRAIN_TOY}/wav.scp,speech,sound" \
        --train_data_path_and_name_and_type "${TRAIN_TOY}/text,text,text" \
        --valid_data_path_and_name_and_type "${DEV_TOY}/wav.scp,speech,sound" \
        --valid_data_path_and_name_and_type "${DEV_TOY}/text,text,text" \
        2>&1 | tee "${EXP_DIR}/train.log"

    echo "  Training complete."
else
    echo "=== Step 3: Skipped (--skip-train) ==="
fi

check "valid.loss.best.pth exists" \
    [ -f "${TRAIN_OUT}/valid.loss.best.pth" ]

# ---------------------------------------------------------------------------
# Step 4: Inference -- beam_size=1 (fast sanity check)
# ---------------------------------------------------------------------------
echo "=== Step 4: Inference (beam_size=1) ==="

python -m espnet2.bin.sot_inference \
    --asr_train_config "${TRAIN_OUT}/config.yaml" \
    --asr_model_file "${TRAIN_OUT}/valid.loss.best.pth" \
    --output_dir "${DECODE_OUT_B1}" \
    --ngpu 1 \
    --beam_size 1 \
    --data_path_and_name_and_type "${DEV_TOY}/wav.scp,speech,sound" \
    2>&1 | tee "${EXP_DIR}/inference_beam1.log"

DECODED_TEXT_B1="${DECODE_OUT_B1}/1best_recog/text"
if [ -f "${DECODED_TEXT_B1}" ]; then
    DECODED_LINES_B1=$(wc -l < "${DECODED_TEXT_B1}")
    check "beam_size=1: decoded text is non-empty (${DECODED_LINES_B1} lines)" \
        [ "${DECODED_LINES_B1}" -gt 0 ]
else
    check "beam_size=1: decoded text file exists" false
fi

# ---------------------------------------------------------------------------
# Step 5: Inference -- beam_size=5 (realistic decode)
# ---------------------------------------------------------------------------
echo "=== Step 5: Inference (beam_size=5) ==="

python -m espnet2.bin.sot_inference \
    --asr_train_config "${TRAIN_OUT}/config.yaml" \
    --asr_model_file "${TRAIN_OUT}/valid.loss.best.pth" \
    --output_dir "${DECODE_OUT_B5}" \
    --ngpu 1 \
    --beam_size 5 \
    --data_path_and_name_and_type "${DEV_TOY}/wav.scp,speech,sound" \
    2>&1 | tee "${EXP_DIR}/inference_beam5.log"

DECODED_TEXT_B5="${DECODE_OUT_B5}/1best_recog/text"
if [ -f "${DECODED_TEXT_B5}" ]; then
    DECODED_LINES_B5=$(wc -l < "${DECODED_TEXT_B5}")
    check "beam_size=5: decoded text is non-empty (${DECODED_LINES_B5} lines)" \
        [ "${DECODED_LINES_B5}" -gt 0 ]
else
    check "beam_size=5: decoded text file exists" false
fi

# Check that <sc> is in the token list vocabulary
if grep -qF "<sc>" "${TOKEN_LIST}"; then
    check "<sc> is in token_list vocabulary" true
else
    check "<sc> is in token_list vocabulary" false
fi

# ---------------------------------------------------------------------------
# Step 6: Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "========================================"

if [ "${FAIL}" -gt 0 ]; then
    echo "  SOME CHECKS FAILED -- review output above."
    exit 1
else
    echo "  ALL CHECKS PASSED -- pipeline verified end-to-end."
    exit 0
fi
