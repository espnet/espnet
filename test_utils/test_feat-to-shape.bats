#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    # Create an ark for dummy feature
    python << EOF
import h5py
import kaldiio
import numpy as np
import scipy.io.wavfile as W

d = {'A-utt1': np.random.randn(1, 100).astype(np.float32),
     'A-utt2': np.random.randn(300, 2).astype(np.float32),
     'B-utt1': np.random.randn(10, 32).astype(np.float32),
     'B-utt2': np.random.randn(10, 10).astype(np.float32)}

with open('${tmpdir}/feats.ark','wb') as f, h5py.File('${tmpdir}/feats.h5','w') as fh:
    for k in sorted(d):
        v = d[k]
        kaldiio.save_ark(f, {k: v})
        fh[k] = v

with open('${tmpdir}/wav.scp','w') as f:
    for k, v in d.items():
        f.write('{k} ${tmpdir}/{k}.wav\n'.format(k=k))
        W.write('${tmpdir}/{k}.wav'.format(k=k), 8000, v.astype(np.int16))
EOF


    cat << EOF > ${tmpdir}/valid.txt
A-utt1 1,100
A-utt2 300,2
B-utt1 10,32
B-utt2 10,10
EOF
}

teardown() {
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

@test "feat-to-shape.py: --filetype=sound" {

    python ${utils}/feat-to-shape.py --filetype sound scp:${tmpdir}/wav.scp ${tmpdir}/shape.txt
    diff ${tmpdir}/shape.txt ${tmpdir}/valid.txt
}

