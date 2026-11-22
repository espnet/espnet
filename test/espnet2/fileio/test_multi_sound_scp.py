from pathlib import Path

import numpy as np
import soundfile

from espnet2.fileio.multi_sound_scp import MultiSoundScpReader


def test_MultiSoundScpReader(tmp_path: Path):
    audio_path1 = tmp_path / "a1.wav"
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path2 = tmp_path / "a2.wav"
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path3 = tmp_path / "a3.wav"
    audio3 = np.random.randint(-100, 100, 8, dtype=np.int16)

    soundfile.write(audio_path1, audio1, 16)
    soundfile.write(audio_path2, audio2, 16)
    soundfile.write(audio_path3, audio3, 16)

    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write(f"abc {audio_path1}\n")
        f.write(f"def {audio_path2} {audio_path3}\n")

    target = MultiSoundScpReader(p, dtype="int16", pad=0)

    rate, array = target["abc"]
    assert rate == 16
    np.testing.assert_array_equal(array, audio1[None])

    # The shorter audio is right-padded to the length of the longer one
    rate, array = target["def"]
    assert rate == 16
    np.testing.assert_array_equal(
        array, np.stack([audio2, np.pad(audio3, (0, 8))], axis=0)
    )

    assert len(target) == 2
    assert "abc" in target
    assert "def" in target
    assert "zzz" not in target
    assert tuple(target.keys()) == ("abc", "def")
    assert tuple(target) == ("abc", "def")
    assert target.get_path("abc") == [str(audio_path1)]
    assert target.get_path("def") == [str(audio_path2), str(audio_path3)]
