#!/usr/bin/env bats

setup() {
    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import h5py
import kaldi_io_py
import numpy as np

d = {k: np.random.randn(100, 100).astype(np.float32)
     for k in ['A-utt1', 'A-utt2', 'B-utt1', 'B-utt2']}

with open('${tmpdir}/feats.ark','wb') as f, h5py.File('${tmpdir}/feats.h5','w') as fh:
    for k, v in d.items():
        kaldi_io_py.write_mat(f, v, key=k)
        fh[k] = v
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

@test "Calc global cmvn stats: --in-filetype=mat" {
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

@test "Calc global cmvn stats: --in-filetype=hdf5" {
    if ! which compute-cmvn-stats &> /dev/null; then
        skip
    fi

    python ${utils}/compute-cmvn-stats.py --in-filetype hdf5 ark:${tmpdir}/feats.h5 ${tmpdir}/test.mat
    compute-cmvn-stats ark:${tmpdir}/feats.ark ${tmpdir}/valid.mat
    python << EOF
import numpy as np
import kaldi_io_py
test = kaldi_io_py.read_mat('${tmpdir}/test.mat')
valid = kaldi_io_py.read_mat('${tmpdir}/valid.mat')
np.testing.assert_allclose(test, valid, rtol=1e-4)
EOF
}


