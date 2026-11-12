#!/usr/bin/env bash
# Run the TSE smoke recipe in a disposable clone, never in a corpus recipe.
set -euo pipefail

. tools/activate_python.sh
. tools/extra_path.sh

python="coverage run --append"
repo_root=$(pwd)
export COVERAGE_FILE="${repo_root}/.coverage"
fixture_root=$(mktemp -d)
trap 'rm -rf "${fixture_root}"' EXIT

espnet3 clone mini_an4/esp2_tse --project "${fixture_root}/recipe"
cp egs3/mini_an4/esp2_asr/downloads.tar.gz "${fixture_root}/recipe/"
cd "${fixture_root}/recipe"
${python} run.py \
    --stages create_dataset collect_stats train infer measure pack_model pack_demo \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    --publication_config conf/publication.yaml \
    --demo_config conf/demo.yaml
cd "${fixture_root}"
${python} "${repo_root}/ci/check_espnet3_tse.py" --recipe "${fixture_root}/recipe"
