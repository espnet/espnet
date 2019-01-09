#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import kaldiio

d = {'A-utt1': np.random.randn(1, 100).astype(np.float32),
     'A-utt2': np.random.randn(20, 100).astype(np.float32),
     'B-utt1': np.random.randn(100, 100).astype(np.float32),
     'B-utt2': np.random.randn(10, 100).astype(np.float32)}

with open('${tmpdir}/feats.ark','wb') as f:
    kaldiio.save_ark(f, d)

count = sum(len(v) for v in d.values())
sums = sum(v.sum(axis=0) for v in d.values())
square_sums = sum((v ** 2).sum(axis=0) for v in d.values())

stats = np.empty((2, len(sums) + 1), dtype=np.float64)
stats[0, :-1] = sums
stats[0, -1] = count
stats[1, :-1] = square_sums
stats[1, -1] = 0
kaldiio.save_mat('${tmpdir}/stats.mat', stats)

EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "apply-cmvn.py" {
    if ! which apply-cmvn &> /dev/null; then
        skip
    fi

    python ${utils}/apply-cmvn.py ${tmpdir}/stats.mat ark:${tmpdir}/feats.ark ark:${tmpdir}/test.ark
    apply-cmvn ${tmpdir}/stats.mat ark:${tmpdir}/feats.ark ark:${tmpdir}/valid.ark
    python << EOF
import numpy as np
import kaldiio
test = dict(kaldiio.load_ark('${tmpdir}/test.ark'))
valid = dict(kaldiio.load_ark('${tmpdir}/valid.ark'))
for k in test:
    np.testing.assert_allclose(test[k], valid[k], rtol=1e-3)
EOF

}


