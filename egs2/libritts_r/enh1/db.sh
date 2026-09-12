# egs2/libritts_r/enh1/db.sh
# Dataset and noise source paths for the ESPnet-Sidon recipe.
# Set paths to existing directories; leave "" to skip.
#
# SR information is NOT stored here — it lives in local/data.sh.
# To add a new dataset: add DATASET_* here + SR entry in local/data.sh.

# ── Clean speech datasets ─────────────────────────────────────────────────
DATASET_LIBRITTS_R=""   # 585h, 24kHz, en  (mandatory)
DATASET_JVS=""                       # 24h,  24kHz, ja
DATASET_FLEURS_R=""                  # 1.3kh,24kHz, 102 langs
DATASET_VCTK_DEMAND=""        # 44h,  48kHz, en
DATASET_EARS=""                      # 100h, 48kHz, en
DATASET_EXPRESSO=""                  # 40h,  48kHz, en
DATASET_HIFICAPTAIN=""               # 36h,  48kHz, ja+en
DATASET_JSUT=""                      # 10h,  48kHz, ja
DATASET_BIBLETTS=""                  # 80h,  48kHz, multilingual

# Required for stage 1 test-set preparation and stages 6--8 evaluation.
# ── Test data: LibriTTS ORIGINAL (inference input, not restored) ──────────
LIBRITTS=""

# ── Noise sources for degradation simulation ─────────────────────────────
NOISE_WHAM=""  # WHAM! noise
NOISE_AUDIOSET=""                    # AudioSet noise clips
NOISE_FSD50K=""                      # FSD50K
NOISE_SCWIND=""                      # SC-Wind synthetic wind noise
