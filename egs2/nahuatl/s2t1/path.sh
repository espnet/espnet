RECIPE_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ESPNET_ROOT=$(realpath "$RECIPE_ROOT/../../..")
VENV_ROOT=$(realpath "$ESPNET_ROOT/../.venv")

# shellcheck disable=SC1091
source "$VENV_ROOT/bin/activate"

export PYTHONPATH="$ESPNET_ROOT:${PYTHONPATH:-}"

# ESPnet/Kaldi helper scripts (run.pl, split_scp.pl, ...) live in utils/
export PATH="$RECIPE_ROOT/utils:$PATH"

# Path to the Nahuatl HF dataset comes from db.sh (the NAHUATL corpus entry),
# like every other corpus location in egs2/.

# Absolute path to the model download cache (OWSM checkpoint + assets)
export MODEL_CACHE_DIR="$ESPNET_ROOT/../model_cache"
mkdir -p "$MODEL_CACHE_DIR"
