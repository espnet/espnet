#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import numpy as np
import scipy.io.wavfile as W

for idx in range(4):
    W.write('${tmpdir}/feats.{}.wav'.format(idx),
            8000, np.random.randint(-100, 100, 100, dtype=np.int16))
with open('${tmpdir}/wav.scp', 'w') as f:
    for idx in range(4):
        f.write('utt{idx} ${tmpdir}/feats.{idx}.wav\n'.format(idx=idx))
EOF

}

teardown() {
    rm -rf $tmpdir
}

@test "dump-pcm.py" {

    python ${utils}/dump-pcm.py --filetype hdf5 scp:${tmpdir}/wav.scp ark:${tmpdir}/feats.h5
    python << EOF
import h5py
import numpy as np
import scipy.io.wavfile as W

with h5py.File('${tmpdir}/feats.h5') as h, open('${tmpdir}/wav.scp', 'r') as s:
    for line in s:
        key, path = line.strip().split()
        valid, rate = W.read(path)
        test = h[key]
        assert rate == 8000
        np.testing.assert_array_equal(test, valid)
EOF
}


