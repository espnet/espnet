#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import h5py
import kaldi_io_py
import numpy as np

d = {'A-utt1': np.random.randn(1, 100).astype(np.float32),
     'A-utt2': np.random.randn(300, 1).astype(np.float32),
     'B-utt1': np.random.randn(10, 32).astype(np.float32),
     'B-utt2': np.random.randn(10, 10).astype(np.float32)}

with open('${tmpdir}/feats.ark','wb') as f, h5py.File('${tmpdir}/feats.h5','w') as fh:
    for k, v in d.items():
        kaldi_io_py.write_mat(f, v, key=k)
        fh[k] = v
EOF


    cat << EOF > ${tmpdir}/valid.txt
A-utt1 1,100
A-utt2 300,1
B-utt1 10,32
B-utt2 10,10
EOF
}

teardown() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    rm -rf $tmpdir
}

@test "feat-to-shape.py: --filetype=mat" {

    python ${utils}/feat-to-shape.py ark:${tmpdir}/feats.ark ${tmpdir}/shape.txt
    diff ${tmpdir}/shape.txt ${tmpdir}/valid.txt
}

@test "feat-to-shape.py: --filetype=hdf5" {

    python ${utils}/feat-to-shape.py --filetype hdf5 ark:${tmpdir}/feats.h5 ${tmpdir}/shape.txt
    diff ${tmpdir}/shape.txt ${tmpdir}/valid.txt
}

