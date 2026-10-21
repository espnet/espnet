#!/usr/bin/env bash
# Select corpora explicitly; each preparer reuses the corresponding ESPnet recipe.
set -euo pipefail
datasets="voxlingua107"
. utils/parse_options.sh
for dataset in ${datasets//,/ }; do
    case "${dataset}" in
        voxlingua107|fleurs|ml_superb2|babel|voxpopuli)
            bash "local/prepare_${dataset}.sh"
            if [ "${dataset}" = voxlingua107 ]; then
                python3 local/prepare_lang2vec.py \
                    --lang2utt data/train_voxlingua107_lang/lang2utt \
                    --output data/voxlingua107_lang2vec.txt
            fi
            ;;
        *) echo "Unknown GeoLID dataset: ${dataset}" >&2; exit 1 ;;
    esac
done
