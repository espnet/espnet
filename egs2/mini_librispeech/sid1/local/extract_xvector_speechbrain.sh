#!/bin/bash
  
# Copyright 2023 Carnegie Mellon University (Jiatong Shi)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

mkdir -p dump/extracted_speechbrain


for dset in train dev test; do
    datadir=dump/raw/${dset}
    outdir=dump/extracted_speechbrain/${dset}
    cp -r ${datadir} ${outdir}
    python3 pyscripts/utils/extract_xvectors.py --toolkit speechbrain \
       --keep_sequence true --pretrained_model speechbrain/spkrec-xvect-voxceleb \
       ${datadir} ${outdir}

    cp ${outdir}/xvector.scp ${outdir}/feats.scp
    echo "512" > ${outdir}/feats_dim
    echo "xvector" > ${outdir}/feats_type
done

