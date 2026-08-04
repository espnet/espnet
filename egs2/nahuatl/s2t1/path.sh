RECIPE_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ESPNET_ROOT=$(realpath "$RECIPE_ROOT/../../..")
VENV_ROOT=$(realpath "$ESPNET_ROOT/../.venv")

# shellcheck disable=SC1091
source "$VENV_ROOT/bin/activate"

export PYTHONPATH="$ESPNET_ROOT:${PYTHONPATH:-}"

# ESPnet/Kaldi helper scripts (run.pl, split_scp.pl, ...) live in utils/
export PATH="$RECIPE_ROOT/utils:$PATH"

# Absolute path to hf_data (read-only source of truth)
export HF_DATA_DIR="$ESPNET_ROOT/../hf_data"

# Absolute path to the model download cache
export MODEL_CACHE_DIR="$ESPNET_ROOT/../model_cache"
mkdir -p "$MODEL_CACHE_DIR"
