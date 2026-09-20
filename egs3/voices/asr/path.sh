#!/usr/bin/env bash
voices_recipe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${voices_recipe_dir}/../../..:${PYTHONPATH:-}"
if [ -d "${voices_recipe_dir}/../../../tools/sctk/bin" ]; then
    export PATH="${voices_recipe_dir}/../../../tools/sctk/bin:${PATH}"
fi
