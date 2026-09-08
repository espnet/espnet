#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# The wavlab-speech/versa commit that the dialog_eval helpers in
# egs2/TEMPLATE/asr1/pyscripts/utils/dialog_eval were last checked against.
# A plain clone tracks HEAD, which is how those imports drifted once already
# (espnet/espnet#6618). Override to follow a branch or another commit:
#   VERSA_COMMIT=main ./installers/install_versa.sh
VERSA_COMMIT="${VERSA_COMMIT:-4f1014c9990e0bda1e21ef58390e4e25d220f768}"

rm -rf versa

git clone https://github.com/wavlab-speech/versa.git
cd versa
git checkout --quiet "${VERSA_COMMIT}"
pip install -e ".[audio]"
cd ..
