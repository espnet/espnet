#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import kaldi_io_py
with open('${tmpdir}/feats.ark','wb') as f:
    kaldi_io_py.write_mat(f, np.random.randn(1, 100).astype(np.float32), key='A-utt1')
    kaldi_io_py.write_mat(f, np.random.randn(20, 100).astype(np.float32), key='A-utt2')
    kaldi_io_py.write_mat(f, np.random.randn(100, 100).astype(np.float32), key='B-utt1')
    kaldi_io_py.write_mat(f, np.random.randn(10, 100).astype(np.float32), key='B-utt2')
EOF

}

teardown() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    rm -rf $tmpdir
}

@test "copy-feats.py: write ark" {
    if ! which copy-feats &> /dev/null; then
        skip
    fi

    python ${utils}/copy-feats.py ark:${tmpdir}/feats.ark ark:${tmpdir}/test.ark
    copy-feats ark:${tmpdir}/feats.ark ark:${tmpdir}/valid.ark
    diff ${tmpdir}/test.ark ${tmpdir}/valid.ark
}


@test "copy-feats.py: write scp" {
    if ! which copy-feats &> /dev/null; then
        skip
    fi

    python ${utils}/copy-feats.py ark:${tmpdir}/feats.ark ark,scp:${tmpdir}/dummy.ark,${tmpdir}/test.scp
    copy-feats ark:${tmpdir}/feats.ark ark,scp:${tmpdir}/dummy.ark,${tmpdir}/valid.scp
    diff ${tmpdir}/test.scp ${tmpdir}/valid.scp
}

@test "copy-feats.py: ark -> hdf5 -> ark" {
    if ! which copy-feats &> /dev/null; then
        skip
    fi

    python ${utils}/copy-feats.py --out-filetype hdf5 ark:${tmpdir}/feats.ark ark:${tmpdir}/feats.h5
    python ${utils}/copy-feats.py --in-filetype hdf5 ark:${tmpdir}/feats.h5 ark:${tmpdir}/test.ark
    diff ${tmpdir}/test.ark ${tmpdir}/feats.ark
}
