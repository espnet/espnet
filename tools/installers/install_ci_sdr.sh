#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_version=$(python3 -c "import torch; print(torch.__version__)")
echo "[INFO] torch_version=${torch_version}"

# ci_sdr is also in espnet[enh]; this installer is for environments built
# with tools/Makefile. >=0.0.3: 0.0.2 declares a git URL dependency, which
# pip>=26.3 refuses (fixed by https://github.com/fgnt/ci_sdr/pull/8).

# ci_sdr depends on torch: keep the one this environment installed, as
# install_torcheval.sh does.
cat >> ci_sdr_constraints.txt << EOF
torch==${torch_version}
EOF

python3 -m pip install -c ci_sdr_constraints.txt \
    "ci_sdr>=0.0.3"

current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ "${torch_version}" != "${current_torch_version}" ]; then
    echo "[ERROR] installing ci_sdr changed torch from ${torch_version} to" \
         "${current_torch_version}. Please report to espnet developers" >&2
    exit 1
fi
