#!/usr/bin/env bats

setup() {
    [ ! -z $LC_ALL ] && export LC_ALL="en_US.UTF-8"

    utils=$(cd $BATS_TEST_DIRNAME/..; pwd)/utils
    tmpdir=$(mktemp -d testXXXXXX)

    cat << EOF > ${tmpdir}/a.scp
uttid1 a1.wav
uttid2 a2.wav
uttid3 a3.wav
uttid4 a4.wav
EOF

    cat << EOF > ${tmpdir}/b.scp
uttid1 b1.wav
uttid2 b2.wav
uttid3 b3.wav
uttid4 b4.wav
EOF

    cat << EOF > ${tmpdir}/c.scp
uttid1 c1.wav
uttid2 c2.wav
uttid3 c3.wav
uttid4 c4.wav
EOF

    cat << EOF > ${tmpdir}/valid.scp
uttid1 sox -M a1.wav b1.wav c1.wav -c 3 -t wav - |
uttid2 sox -M a2.wav b2.wav c2.wav -c 3 -t wav - |
uttid3 sox -M a3.wav b3.wav c3.wav -c 3 -t wav - |
uttid4 sox -M a4.wav b4.wav c4.wav -c 3 -t wav - |
EOF


}

teardown() {
    rm -rf $tmpdir
}

@test "mix-mono-wav-scp.py" {
    python $utils/mix-mono-wav-scp.py ${tmpdir}/a.scp ${tmpdir}/b.scp ${tmpdir}/c.scp > ${tmpdir}/out.scp
    diff ${tmpdir}/out.scp ${tmpdir}/valid.scp
}

