from pathlib import Path

from espnet2.fileio.vad_scp import VADScpReader, VADScpWriter


def test_VADScpReader(tmp_path: Path):
    p = tmp_path / "vad.scp"
    with p.open("w") as f:
        f.write("abc 0.0000:1.2000\n")
        f.write("def 3.0000:4.5000 7.0000:9.0000\n")

    desired = {"abc": [(0.0, 1.2)], "def": [(3.0, 4.5), (7.0, 9.0)]}
    target = VADScpReader(p)

    for k in desired:
        assert target[k] == desired[k]
    assert len(target) == len(desired)
    assert tuple(target.keys()) == tuple(desired)
    assert "zzz" not in target


def test_VADScpWriter(tmp_path: Path):
    p = tmp_path / "vad.scp"
    with VADScpWriter(p) as writer:
        writer["abc"] = [(0.0, 1.2)]
        writer["def"] = [(3.0, 4.5), (7.0, 9.0)]

    assert p.read_text() == "abc 0.0000:1.2000\ndef 3.0000:4.5000 7.0000:9.0000\n"
    target = VADScpReader(p)
    assert target["abc"] == [(0.0, 1.2)]
    assert target["def"] == [(3.0, 4.5), (7.0, 9.0)]
