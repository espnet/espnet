#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi
torch_version=$(python3 -c "import torch; print(torch.__version__)")
# Drop the local-version segment (e.g. "2.9.1+cpu" -> "2.9.1") so the
# constraint is a public PEP 440 version. Otherwise pip's resolver
# rejects every lightning candidate because their `torch >=X, <Y`
# requirement cannot select a local version, producing
# "No matching distribution found for lightning". The post-install
# assertion below still catches any wheel swap.
torch_version_public=${torch_version%+*}

echo "[INFO] torch_version=${torch_version} (constraint: ${torch_version_public})"

cat >> lightning_constraints.txt << EOF
torch==${torch_version_public}
EOF

python3 -m pip install -c lightning_constraints.txt lightning

# Check the pytorch version is not changed from the original version
current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ ${torch_version} != "${current_torch_version}" ]; then
    echo "[ERROR] The torch version has been changed. Please report to espnet developers"
    exit 1
fi
