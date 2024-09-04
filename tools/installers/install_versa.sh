#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

rm -rf versa

# VERSA   Commit id when making this PR: `commit 3e73ea43659baffb5cd42a8e5946cb659ad27535`
git clone https://github.com/shinjiwlab/versa.git
cd versa
pip install -e .
cd ..
