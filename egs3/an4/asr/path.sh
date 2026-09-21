#!/usr/bin/env bash
# Source this file after activating an environment with ESPnet dependencies.
an4_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${an4_repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
unset an4_repo_root
