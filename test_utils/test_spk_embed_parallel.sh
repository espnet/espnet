# How to run:
# REPO_ROOT=$PWD CUDA_VISIBLE_DEVICES= TOOLKIT=espnet \
# SPK_EMBED_TAG=xvector PRETRAINED_MODEL=espnet/voxcelebs12_xvector_mel \
# NJ_PAR=4 KEEP_TMP=0 \
# bash test_utils/test_spk_embed_parallel.sh


set -euo pipefail

# ----------------------------- Config -----------------------------
REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
SCRIPT_SEQ="$REPO_ROOT/egs2/TEMPLATE/asr1/pyscripts/utils/extract_spk_embed.py"
SCRIPT_PAR="$REPO_ROOT/egs2/TEMPLATE/asr1/pyscripts/utils/extract_spk_embed_parallel.py"
SORT_SCRIPT="$REPO_ROOT/egs2/TEMPLATE/tts1/scripts/utils/sort_spk_embed_scp.sh"

TOOLKIT=${TOOLKIT:-espnet}                 # espnet | speechbrain | rawnet
# If you have a local ECAPA .pth, set PRETRAINED_MODEL=/path/to/valid.acc.best.pth and keep SPK_EMBED_TAG=ecapa
PRETRAINED_MODEL=${PRETRAINED_MODEL:-espnet/voxcelebs12_xvector_mel}
SPK_EMBED_TAG=${SPK_EMBED_TAG:-ecapa}
NJ_SEQ=${NJ_SEQ:-1}
NJ_PAR=${NJ_PAR:-2}
SR=16000
DUR=0.5                                     # seconds per utterance
KEEP_TMP=${KEEP_TMP:-0}

echo "Using:"
echo "  REPO_ROOT       = $REPO_ROOT"
echo "  TOOLKIT         = $TOOLKIT"
echo "  SPK_EMBED_TAG   = $SPK_EMBED_TAG"
echo "  PRETRAINED_MODEL= $PRETRAINED_MODEL"
echo

# Sanity: scripts present?
[[ -f "$SCRIPT_SEQ" ]] || { echo "Missing $SCRIPT_SEQ"; exit 1; }
[[ -f "$SCRIPT_PAR" ]] || { echo "Missing $SCRIPT_PAR"; exit 1; }

# --------------------------- Workspace ---------------------------
TMPDIR=$(mktemp -d -t spkembtest.XXXXXXXX)
trap '[[ "$KEEP_TMP" == 1 ]] || rm -rf "$TMPDIR"' EXIT

DATA_DIR="$TMPDIR/data"
OUT_SEQ="$TMPDIR/out_seq"
OUT_PAR="$TMPDIR/out_par"
mkdir -p "$DATA_DIR" "$OUT_SEQ" "$OUT_PAR"

# -------------------- Create tiny synthetic data ------------------
# 4 utterances, 2 speakers, pure tones with different freqs
python3 - "$DATA_DIR" <<'PY'
import os, wave, struct, math
import pathlib

root = pathlib.Path(os.sys.argv[1])
audio = root
os.makedirs(audio, exist_ok=True)

sr = 16000
DUR = 0.5
N = int(sr * DUR)

spec = {
    "spk1_utt1": 440.0,
    "spk1_utt2": 660.0,
    "spk2_utt1": 880.0,
    "spk2_utt2": 990.0,
}

wav_scp = []
utt2spk = []
text = []

for utt, freq in spec.items():
    spk = utt.split('_')[0]
    path = audio / f"{utt}.wav"
    # generate 16-bit PCM sine wave
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for n in range(N):
            v = int(0.3 * 32767 * math.sin(2*math.pi*freq*(n/sr)))
            wf.writeframes(struct.pack('<h', v))
    wav_scp.append(f"{utt} {path}\n")
    utt2spk.append(f"{utt} {spk}\n")
    text.append(f"{utt} hello_{utt}\n")

(open(root/"wav.scp", 'w')).writelines(wav_scp)
(open(root/"utt2spk", 'w')).writelines(utt2spk)
(open(root/"text", 'w')).writelines(text)

# build spk2utt
spk2utt_map = {}
for line in utt2spk:
    u, s = line.strip().split()
    spk2utt_map.setdefault(s, []).append(u)
with open(root/"spk2utt", 'w') as f:
    for s, us in sorted(spk2utt_map.items()):
        f.write(s + ' ' + ' '.join(us) + '\n')
PY

printf "\nCreated toy data in: %s\n\n" "$DATA_DIR"
cat "$DATA_DIR/wav.scp"; echo
cat "$DATA_DIR/utt2spk"; echo
cat "$DATA_DIR/spk2utt"; echo

# -------------------- Run sequential extractor -------------------
export CUDA_VISIBLE_DEVICES=""  # force CPU

echo "[1/4] Running SEQUENTIAL extractor ($TOOLKIT:$SPK_EMBED_TAG)..."
python3 "$SCRIPT_SEQ" \
  --toolkit "$TOOLKIT" \
  --spk_embed_tag "$SPK_EMBED_TAG" \
  --pretrained_model "$PRETRAINED_MODEL" \
  --device cpu \
  "$DATA_DIR" "$OUT_SEQ"

# --------------------- Run parallel extractor --------------------
echo "[2/4] Running PARALLEL extractor ($TOOLKIT:$SPK_EMBED_TAG, workers=$NJ_PAR)..."
python3 "$SCRIPT_PAR" \
  --toolkit "$TOOLKIT" \
  --spk_embed_tag "$SPK_EMBED_TAG" \
  --pretrained_model "$PRETRAINED_MODEL" \
  --device cpu \
  --num_workers "$NJ_PAR" \
  "$DATA_DIR" "$OUT_PAR" \
  --batch_size "${SPK_EMBED_BATCH:-8}" \
  --prefetch "${SPK_EMBED_PREFETCH:-64}" \



# ---------------------- (Optional) Sorting step ------------------
if [[ -f "$SORT_SCRIPT" && -f "$OUT_PAR/utt_emb.scp" ]]; then
  echo "[3/4] Sorting UTT embeddings to match text (sanity check) ..."
  bash "$SORT_SCRIPT" \
    --data-dir "$DATA_DIR" \
    --spk-embed-scp "$OUT_PAR/utt_emb.scp" \
    --out-scp "$OUT_PAR/utt_emb.sorted.scp"
else
  echo "[3/4] Sort step skipped (script or utt_emb.scp not found)"
fi

# -------------------------- Compare outputs ----------------------
echo "[4/4] Comparing embeddings (seq vs par) ..."
echo "[4/4] Comparing embeddings (seq vs par) ..."
python3 - "$DATA_DIR" "$OUT_SEQ" "$OUT_PAR" <<'PY'
import os, glob, sys, numpy as np
from kaldiio import ReadHelper

DATA, OUT_SEQ, OUT_PAR = sys.argv[1:4]
utts = [l.split()[0] for l in open(os.path.join(DATA, "wav.scp"))]
spks = [l.split()[0] for l in open(os.path.join(DATA, "spk2utt"))]

def load_scp(path):
    d = {}
    with ReadHelper(f"scp:{path}") as r:
        for k, v in r: d[k] = np.asarray(v)
    return d

def pick_scp(dirpath, want_keys):
    cands = sorted(glob.glob(os.path.join(dirpath, "*.scp")))
    preferred = [p for p in cands if any(x in os.path.basename(p) for x in ("utt","embed","xvector","spk"))]
    for pool in (preferred, cands):
        for scp in pool:
            try:
                d = load_scp(scp)
            except Exception:
                continue
            if set(want_keys).issubset(d.keys()):
                return scp
    raise SystemExit(f"No .scp in {dirpath} contains all keys: {want_keys}")

utt_seq = pick_scp(OUT_SEQ, utts)
utt_par = pick_scp(OUT_PAR, utts)
spk_seq = pick_scp(OUT_SEQ, spks)
spk_par = pick_scp(OUT_PAR, spks)

print("UTT SEQ:", utt_seq)
print("UTT PAR:", utt_par)
print("SPK SEQ:", spk_seq)
print("SPK PAR:", spk_par)

def compare_utt(A_scp, B_scp, keys, rtol=1e-5, atol=1e-5):
    A, B = load_scp(A_scp), load_scp(B_scp)
    max_abs = 0.0
    for k in keys:
        a, b = A[k], B[k]
        if a.shape != b.shape:
            raise SystemExit(f"[UTT] shape mismatch for {k}: {a.shape} vs {b.shape}")
        diff = float(np.max(np.abs(a - b)))
        max_abs = max(max_abs, diff)
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise SystemExit(f"[UTT] {k} differ; max_abs_diff={diff:.6g}")
    print(f"[UTT] OK: {len(keys)} items match; max_abs_diff={max_abs:.3g}")

def compare_spk(A_scp, B_scp, keys, rtol=1e-5, atol=1e-5):
    A, B = load_scp(A_scp), load_scp(B_scp)
    max_abs = 0.0
    for k in keys:
        a, b = A[k], B[k]
        # Accept either (512,) or (1,512) etc. — squeeze singleton dims.
        a = np.squeeze(a)
        b = np.squeeze(b)
        if a.ndim != 1 or b.ndim != 1:
            raise SystemExit(f"[SPK] unexpected ndim for {k}: {a.shape} vs {b.shape}")
        if a.shape != b.shape:
            raise SystemExit(f"[SPK] shape mismatch for {k}: {a.shape} vs {b.shape}")
        diff = float(np.max(np.abs(a - b)))
        max_abs = max(max_abs, diff)
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise SystemExit(f"[SPK] {k} differ; max_abs_diff={diff:.6g}")
    print(f"[SPK] OK: {len(keys)} items match; max_abs_diff={max_abs:.3g}")

compare_utt(utt_seq, utt_par, utts)
compare_spk(spk_seq, spk_par, spks)
PY
