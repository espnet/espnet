#!/usr/bin/env bash
# Source this file after activating an environment with ESPnet dependencies.
an4_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${an4_repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -d "${an4_repo_root}/tools/sctk/bin" ]; then
    export PATH="${an4_repo_root}/tools/sctk/bin:${PATH}"
fi
unset an4_repo_root
