#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_version=$(python3 -c "import torch; print(torch.__version__)")
echo "[INFO] torch_version=${torch_version}"

# Deliberately not a version pin on torcheval itself. Its newest release is
# 0.0.7, from August 2023, and its metadata declares no dependency on torch, so
# "resolves to whatever is newest" is a risk that does not exist today - while a
# pinned literal here would be one more number nothing watches. dependabot
# covers github-actions and not tools/installers, which is how TH_VERSION sat at
# 2.7.1 for five months and broke the weekly docker publish.
#
# What is guarded is torch being replaced if some future release does declare
# it. install_lightning.sh guards the same thing the same way.
cat >> torcheval_constraints.txt << EOF
torch==${torch_version}
EOF

python3 -m pip install -c torcheval_constraints.txt torcheval

# The constraint is not evidence on its own: check torch is the version it was.
current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ "${torch_version}" != "${current_torch_version}" ]; then
    echo "[ERROR] installing torcheval changed torch from ${torch_version} to" \
         "${current_torch_version}. Please report to espnet developers" >&2
    exit 1
fi
