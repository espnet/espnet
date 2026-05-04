#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
    exit 0
fi

PRIMARY_URL="https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz"
BACKUP_URL="https://huggingface.co/espnet/ci_tools/resolve/main/mwerSegmenter.tar.gz"

if ! wget --no-check-certificate --tries=3 -O mwerSegmenter.tar.gz "${PRIMARY_URL}"; then
    echo "Primary download failed, trying backup URL..."
    echo ""
    echo "=============================================================================="
    echo "LEGAL DISCLAIMER:"
    echo "The backup URL (HuggingFace mirror) is provided only for installation purposes."
    echo "The mwerSegmenter software is subject to the RWTH mwerSegmenter License."
    echo ""
    echo "This software is for NON-COMMERCIAL USE ONLY."
    echo ""
    echo "Key terms of the RWTH mwerSegmenter License:"
    echo "- Non-exclusive rights are granted for non-commercial use only."
    echo "- Any commercial use or distribution requires prior authorization by RWTH."
    echo "- The software is provided 'AS IS' with no warranty of any kind."
    echo "- This license is subject to German law."
    echo ""
    echo "For the full license text, please refer to the original RWTH mwerSegmenter"
    echo "distribution or contact RWTH directly."
    echo "=============================================================================="
    echo ""
    if ! wget --no-check-certificate --tries=3 -O mwerSegmenter.tar.gz "${BACKUP_URL}"; then
        echo "Both primary and backup downloads failed"
        exit 1
    fi
fi
tar zxvf mwerSegmenter.tar.gz
rm mwerSegmenter.tar.gz

patch mwerSegmenter/hyp2sgm.py < installers/patch_mwerSegmenter/hyp2sgm.patch
patch mwerSegmenter/sgm2mref.py < installers/patch_mwerSegmenter/sgm2mref.patch
