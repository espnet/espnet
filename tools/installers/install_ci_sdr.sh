#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_version=$(python3 -c "import torch; print(torch.__version__)")
echo "[INFO] torch_version=${torch_version}"

# From git, not PyPI. ci_sdr's only PyPI release (0.0.2, an sdist) declares
# `padertorch @ git+...` under its own [all] extra, and pip>=26.3 refuses any
# PyPI distribution with a direct URL dependency - so `pip install ci_sdr`
# fails, and so did espnet[enh] while it listed ci_sdr. The upstream fix is
# https://github.com/fgnt/ci_sdr/pull/8; once a fixed release is on PyPI this
# can install that instead.
#
# Pinned to a commit so every environment gets the same code.
ci_sdr_commit=380c8c23bfbd0d77fac83841a64fb8759da9971f

# ci_sdr depends on torch: keep the one this environment installed, as
# install_torcheval.sh does.
cat >> ci_sdr_constraints.txt << EOF
torch==${torch_version}
EOF

python3 -m pip install -c ci_sdr_constraints.txt \
    "ci_sdr @ git+https://github.com/fgnt/ci_sdr@${ci_sdr_commit}"

current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ "${torch_version}" != "${current_torch_version}" ]; then
    echo "[ERROR] installing ci_sdr changed torch from ${torch_version} to" \
         "${current_torch_version}. Please report to espnet developers" >&2
    exit 1
fi
