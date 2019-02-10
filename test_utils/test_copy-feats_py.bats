#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import kaldiio
with open('${tmpdir}/feats.ark','wb') as f:
    kaldiio.save_ark(f, {'A-utt1': np.random.randn(1, 100).astype(np.float32),
                         'A-utt2': np.random.randn(20, 100).astype(np.float32),
                         'B-utt1': np.random.randn(100, 100).astype(np.float32),
                         'B-utt2': np.random.randn(10, 100).astype(np.float32)})
EOF

}

teardown() {
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
