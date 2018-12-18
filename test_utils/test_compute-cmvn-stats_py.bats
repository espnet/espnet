#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import kaldi_io_py
with open('${tmpdir}/feats.ark','wb') as f:
    kaldi_io_py.write_mat(f, np.random.randn(100, 100), key='A-utt1')
    kaldi_io_py.write_mat(f, np.random.randn(100, 100), key='A-utt2')
    kaldi_io_py.write_mat(f, np.random.randn(100, 100), key='B-utt1')
    kaldi_io_py.write_mat(f, np.random.randn(100, 100), key='B-utt2')
EOF

    # Create spk2utt
    cat << EOF > ${tmpdir}/spk2utt
A A-utt1 A-utt2
B B-utt1 B-utt2
EOF
}

teardown() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    rm -rf $tmpdir
}

@test "Calc global cmvn stats" {
    if ! which compute-cmvn-stats &> /dev/null; then
        skip
    fi

    python ${utils}/compute-cmvn-stats.py ark:${tmpdir}/feats.ark ${tmpdir}/test.mat
    compute-cmvn-stats ark:${tmpdir}/feats.ark ${tmpdir}/valid.mat
    python << EOF
import numpy as np
import kaldi_io_py
test = kaldi_io_py.read_mat('${tmpdir}/test.mat')
valid = kaldi_io_py.read_mat('${tmpdir}/valid.mat')
np.testing.assert_allclose(test, valid, rtol=1e-4)
EOF
}


@test "Calc speaker cmvn stats" {
    if ! which compute-cmvn-stats &> /dev/null; then
        skip
    fi

    python ${utils}/compute-cmvn-stats.py --spk2utt ${tmpdir}/spk2utt ark:${tmpdir}/feats.ark ark:${tmpdir}/test.ark
    compute-cmvn-stats --spk2utt=ark:${tmpdir}/spk2utt ark:${tmpdir}/feats.ark ark:${tmpdir}/valid.ark
    python << EOF
import numpy as np
import kaldi_io_py
test = dict(kaldi_io_py.read_mat_ark('${tmpdir}/test.ark'))
valid = dict(kaldi_io_py.read_mat_ark('${tmpdir}/valid.ark'))
for k in test:
    np.testing.assert_allclose(test[k], valid[k], rtol=1e-4)
EOF
}

