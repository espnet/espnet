#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import kaldi_io_py
with open('${tmpdir}/feats.ark','wb') as f:
    kaldi_io_py.write_mat(f, np.random.randn(1, 100), key='A-utt1')
    kaldi_io_py.write_mat(f, np.random.randn(300, 1), key='A-utt2')
    kaldi_io_py.write_mat(f, np.random.randn(10, 32), key='B-utt1')
    kaldi_io_py.write_mat(f, np.random.randn(10, 10), key='B-utt2')
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

@test "feat-to-shape.py" {

    python ${utils}/feat-to-shape.py ark:${tmpdir}/feats.ark ${tmpdir}/shape.txt
    diff ${tmpdir}/shape.txt ${tmpdir}/valid.txt
}

