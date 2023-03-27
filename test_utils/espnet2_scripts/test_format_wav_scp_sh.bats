#!/usr/bin/env bats

setup() {
    export LC_ALL="en_US.UTF-8"

    basedir=$(cd $BATS_TEST_DIRNAME/../..; pwd)/egs2/TEMPLATE/asr1
    cd ${basedir}
    tmpdir=$(mktemp -d testXXXXXX)
}

teardown() {
    cd ${basedir}
    rm -rf $tmpdir
}

@test "convert_with_segments" {
    cd ${basedir}

    python << EOF
import numpy as np
import soundfile

array = np.random.randn(100)
rate = 16000
num = 3

for i in range(num):
    soundfile.write(f"${tmpdir}/{i}.wav", array, rate)
    with open("${tmpdir}/wav.scp", "w") as f:
        f.write(f"{i}_rec ${tmpdir}/{i}.wav\n")
    with open("${tmpdir}/segments", "w") as f:
        f.write(f"{i} {i}_rec 0 {int(100/16000 * 1000)}\n")

EOF

    for audio_format in "wav" "flac"; do
        scripts/audio/format_wav_scp.sh --nj 1 --audio_format ${audio_format} --segments ${tmpdir}/segments ${tmpdir}/wav.scp ${tmpdir}/outs_${audio_format}
        python << EOF
import numpy as np
import soundfile
with open("${tmpdir}/wav.scp", "r") as f, open("${tmpdir}/outs_${audio_format}/wav.scp", "r") as fout:
    for line, line2 in zip(f, fout):
        k, v = line.rstrip().split()
        k2, v2 = line2.rstrip().split()
        array, rate = soundfile.read(v)
        array2, rate2 = soundfile.read(v2)

        assert rate == rate2, (rate, rate2)
        np.testing.assert_equal(array, array2)

EOF
    done

}
@test "read_write_consistency" {
    cd ${basedir}

    python << EOF
import numpy as np
import soundfile

array = np.random.randn(10)
rate = 16000
num = 3

for i in range(num):
    soundfile.write(f"${tmpdir}/{i}.wav", array, rate)
    with open("${tmpdir}/wav.scp", "w") as f:
        f.write(f"{i} ${tmpdir}/{i}.wav\n")

EOF

    for audio_format in "wav" "flac"; do
        scripts/audio/format_wav_scp.sh --nj 1 --audio_format ${audio_format} ${tmpdir}/wav.scp ${tmpdir}/outs_${audio_format}
        python << EOF
import numpy as np
import soundfile
with open("${tmpdir}/wav.scp", "r") as f, open("${tmpdir}/outs_${audio_format}/wav.scp", "r") as fout:
    for line, line2 in zip(f, fout):
        k, v = line.rstrip().split()
        k2, v2 = line2.rstrip().split()
        array, rate = soundfile.read(v)
        array2, rate2 = soundfile.read(v2)

        assert k == k2, (k, k2)
        assert rate == rate2, (rate, rate2)
        np.testing.assert_equal(array, array2)

EOF
    done

}



@test "convert_with_pipe" {
    cd ${basedir}

    python << EOF
import numpy as np
import soundfile

array = np.random.randn(10)
rate = 16000
num = 3

for i in range(num):
    soundfile.write(f"${tmpdir}/{i}.wav", array, rate)
    with open("${tmpdir}/wav.scp", "w") as f:
        f.write(f"{i} ${tmpdir}/{i}.wav\n")

EOF

    for audio_format in "wav" "flac"; do
        scripts/audio/format_wav_scp.sh --nj 1 --audio_format ${audio_format} ${tmpdir}/wav.scp ${tmpdir}/outs_${audio_format}
        python << EOF
import numpy as np
import soundfile
with open("${tmpdir}/wav.scp", "r") as f, open("${tmpdir}/outs_${audio_format}/wav.scp", "r") as fout:
    for line, line2 in zip(f, fout):
        k, v = line.rstrip().split()
        k2, v2 = line2.rstrip().split()
        array, rate = soundfile.read(v)
        array2, rate2 = soundfile.read(v2)

        assert k == k2, (k, k2)
        assert rate == rate2, (rate, rate2)
        np.testing.assert_equal(array, array2)

EOF
    done

}
