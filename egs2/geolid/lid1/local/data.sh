#!/usr/bin/env bash
# DO NOT RUN THIS SCRIPT DIRECTLY!!!!!!!!
log "DO NOT RUN THIS SCRIPT DIRECTLY!!!!!!!!!"
log "Please see the script for each dataset preparation!"
exit 1

# NOTE
# This file is a TEMPLATE that shows the typical order to PREPARE each dataset.
# It’s strongly recommended to run each `prepare_*.sh` script separately
# so issues are easier to locate.
#
# Downloads are NOT handled here. Each dataset must be downloaded/unzipped by you.
# Then open the corresponding `local/prepare_*.sh` and set the correct paths.
#
# We do NOT run this script end-to-end in practice; we prepare datasets one by one.

# 1) Prepare individual datasets (edit paths inside each script before running)
bash local/prepare_babel.sh
bash local/prepare_fleurs.sh
bash local/prepare_ml_superb2.sh
bash local/prepare_voxlingua107.sh
bash local/prepare_voxpopuli.sh

# 2) Combine datasets for training
# Before running combine.sh, make sure each dataset has been fully processed.
# Specifically, run the dump stages (stage 1 -> stage 4) in the lid.sh pipelines,
# e.g., ./run.sh --stage 1 --stop_stage 4.
# After all dumps are ready, run:
bash local/combine.sh
