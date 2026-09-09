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
VERSA_DEFAULT_REF="v1.1.0"
VERSA_REF="${VERSA_REF:-${VERSA_DEFAULT_REF}}"

# The commit VERSA_DEFAULT_REF pointed at when this was written. A tag is not
# immutable - `git tag -f` and a force push retarget it - so on its own the tag
# says what is meant while nothing says what was tested, and a moved tag would
# install different code with the installer still succeeding.
#
# Only checked for the default ref: following a branch or another commit sets
# VERSA_REF and this goes out of the way. Set VERSA_REF_SHA= to skip it.
if [ "${VERSA_REF}" = "${VERSA_DEFAULT_REF}" ]; then
    VERSA_REF_SHA="${VERSA_REF_SHA-38454c95f95d9ea717ebf800abe237e846b0f16e}"
else
    VERSA_REF_SHA="${VERSA_REF_SHA-}"
fi

rm -rf versa

git clone https://github.com/wavlab-speech/versa.git
cd versa
git checkout --quiet "${VERSA_REF}"

# Say which revision this actually is: for an annotated tag the ref resolves
# through a tag object, so "v1.1.0" alone does not identify the tree in a log.
git --no-pager log -1 --format='[INFO] versa %H (%D)'

versa_head="$(git rev-parse HEAD)"
if [ -n "${VERSA_REF_SHA}" ] && [ "${versa_head}" != "${VERSA_REF_SHA}" ]; then
    echo "[ERROR] versa ${VERSA_REF} now points at ${versa_head}," >&2
    echo "        not the ${VERSA_REF_SHA} it was tested against." >&2
    echo "        The tag has been moved. Check what changed, then update" >&2
    echo "        VERSA_REF_SHA in this script - or set VERSA_REF_SHA= to" >&2
    echo "        install it anyway." >&2
    exit 1
fi
pip install -e ".[audio]"
cd ..
