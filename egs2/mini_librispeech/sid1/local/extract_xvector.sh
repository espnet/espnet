#!/bin/bash
  
# Copyright 2023 Carnegie Mellon University (Jiatong Shi)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

mkdir -p dump/extracted

if [ ! -f "model.pt" ]; then
    wget https://huggingface.co/jungjee/RawNet3/resolve/main/model.pt
fi

for dset in train dev test; do
    datadir=dump/raw/${dset}
    outdir=dump/extracted/${dset}
    cp -r ${datadir} ${outdir}
    python3 pyscripts/utils/extract_xvectors.py --toolkit rawnet \
        --pretrained_model model.pt --draw_tsne true --tsne_n_spk 10 \
       ${datadir} ${outdir}

    cp ${outdir}/xvector.scp ${outdir}/feats.scp
    echo "256" > ${outdir}/feats_dim
    echo "xvector" > ${outdir}/feats_type
done

