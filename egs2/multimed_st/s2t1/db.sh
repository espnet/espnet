#!/usr/bin/env bash

# MultiMed-ST local corpus/cache directory.
# Users can override this to point at a shared corpus/cache location.
export MULTIMED_ST=${MULTIMED_ST:-${PWD}/downloads/multimed_st}

# Hugging Face cache directories.
export HF_HOME=${HF_HOME:-${PWD}/downloads/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
