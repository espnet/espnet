#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
git checkout af3078794e01a809d6fa5bd40b95517c67e2df54
pip install -e .

# using mosestokenizer python version in pyscripts/utils/simuleval_agent.py
pip install mosestokenizer
