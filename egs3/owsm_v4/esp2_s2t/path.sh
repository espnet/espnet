#!/usr/bin/env bash
# Corpus locations for this recipe. Values are machine-local, so set them in
# your environment (or in an uncommitted submit/env.sh) before sourcing this.
#
#   SPGISPEECH  root holding train.csv, val.csv and spgispeech/{train,val}/
#   MUST_C      root holding en-<lang>/data/<split>/txt/
#   OWSM_CACHE  where the built caches go; defaults to <recipe>/data/hf.
#               Point it at a large shared filesystem, e.g.
#               export OWSM_CACHE=/shared/scratch/owsm_cache

export PYTHONPATH=../../../:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh

for _var in SPGISPEECH MUST_C; do
    if [ -z "${!_var:-}" ]; then
        echo "path.sh: ${_var} is not set; create_dataset will not find that corpus" >&2
    fi
done
unset _var
