#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    python << EOF
import numpy as np
import scipy.io.wavfile as W

a = np.random.randint(-100, 100, 100, np.int16)
b = np.random.randint(-100, 100, 100, np.int16)
W.write('${tmpdir}/a.wav', 8000, a)
W.write('${tmpdir}/b.wav', 8000, b)
W.write('${tmpdir}/valid.wav', 8000,
        np.concatenate((a[:, None], b[:, None]), axis=1))
EOF


}

teardown() {
    rm -rf $tmpdir
}

@test "gather-wav-scp.py: file" {
    python $utils/gather-wav.py ${tmpdir}/a.wav ${tmpdir}/b.wav > ${tmpdir}/out.wav
    diff ${tmpdir}/out.wav ${tmpdir}/valid.wav
}

@test "gather-wav-scp.py: stream" {
    python $utils/gather-wav.py "cat ${tmpdir}/a.wav |" "cat ${tmpdir}/b.wav |" > ${tmpdir}/out.wav
    diff ${tmpdir}/out.wav ${tmpdir}/valid.wav
}

