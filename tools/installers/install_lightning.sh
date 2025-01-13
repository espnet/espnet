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


# pt_plus(){
#     python3 <<EOF
# import sys
# from packaging.version import parse as L
# if L('$torch_version') >= L('$1'):
#     print("true")
# else:
#     print("false")
# EOF
# }

echo "[INFO] torch_version=${torch_version}"

# # Determine the version of lightning
# # See: https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix
# if $(pt_plus 2.2.0); then
#     lightning_version=2.5
# elif $(pt_plus 2.1.0); then
#     lightning_version=2.4
# elif $(pt_plus 2.0.0); then
#     lightning_version=2.3
# elif $(pt_plus 1.13.0); then
#     lightning_version=2.2
# elif $(pt_plus 1.12.0); then
#     lightning_version=2.1
# elif $(pt_plus 1.11.0); then
#     lightning_version=2.0
# else
#     echo "[ERROR] Our supported lightning requires pytorch>=1.11.0"
#     exit 1;
# fi

# lightning>=${lightning_version},<${lightning_version}.99

cat >> lightning_constraints.txt << EOF
torch==${torch_version}
EOF

python3 -m pip install -c lightning_constraints.txt lightning


# Check the pytorch version is not changed from the original version
current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ ${torch_version} != "${current_torch_version}" ]; then
    echo "[ERROR] The torch version has been changed. Please report to espnet developers"
    exit 1
fi
