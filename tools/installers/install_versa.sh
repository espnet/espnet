#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# The wavlab-speech/versa release the dialog_eval helpers in
# egs2/TEMPLATE/asr1/pyscripts/utils/dialog_eval are checked against. A plain
# clone tracks HEAD, which is how those imports drifted once already
# (espnet/espnet#6618). Override to follow a branch, a tag or a commit:
#   VERSA_REF=main ./installers/install_versa.sh
#
# A tag rather than the commit this used to name. versa had neither tags nor
# releases, so the only thing expressible was "the commit we happened to test"
# (wavlab-speech/versa#92); v1.1.0 is the first release and says what is meant.
# It is three commits ahead of the SHA it replaces, and those three touch only
# .github/workflows/ci.yml, pyproject.toml and versa/__init__.py - no module
# ESPnet imports differs between them.
VERSA_REF="${VERSA_REF:-v1.1.0}"

rm -rf versa

git clone https://github.com/wavlab-speech/versa.git
cd versa
git checkout --quiet "${VERSA_REF}"
# Say which revision this actually is: for an annotated tag the ref resolves
# through a tag object, so "v1.1.0" alone does not identify the tree in a log.
git --no-pager log -1 --format='[INFO] versa %H (%D)'
pip install -e ".[audio]"
cd ..
