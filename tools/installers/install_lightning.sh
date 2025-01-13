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


pt_plus(){
    python3 <<EOF
import sys
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

echo "[INFO] torch_version=${torch_version}"


if $(pt_plus 2.2.0); then
    pip install lightning==2.5.0
elif $(pt_plus 2.1.0); then
    pip install lightning==2.4.0
elif $(pt_plus 2.0.0); then
    pip install lightning==2.3.0
elif $(pt_plus 1.13.0); then
    pip install lightning==2.2.0
elif $(pt_plus 1.12.0); then
    pip install lightning==2.1.0
elif $(pt_plus 1.11.0); then
    pip install lightning==2.0.0
else
    echo "[WARNING] Our supported lightning requires pytorch>=1.11.0"
fi


# Check the pytorch version is not changed from the original version
current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ ${torch_version} != "${current_torch_version}" ]; then
    echo "[ERROR] The torch version has been changed. Please report to espnet developers"
    exit 1
fi
