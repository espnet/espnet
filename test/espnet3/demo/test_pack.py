from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.demo.pack import pack_demo


def test_pack_demo_writes_assets(tmp_path: Path) -> None:
    demo_dir = tmp_path / "packed_demo"
    infer_src = tmp_path / "infer.yaml"
    infer_src.write_text("model: {}\n", encoding="utf-8")
    extra_src = tmp_path / "extra.txt"
    extra_src.write_text("hello\n", encoding="utf-8")
    demo_cfg = OmegaConf.create(
        {
            "system": "asr",
            "infer_config": str(infer_src),
            "output_keys": {"text": "hyp"},
            "pack": {
                "out_dir": str(demo_dir),
                "files": [str(extra_src)],
            },
        }
    )

    class DummySystem:
        demo_config = demo_cfg
        demo_config_path = None
        exp_dir = None
        publish_config = None

    out_dir = pack_demo(DummySystem())
    assert out_dir == demo_dir
    assert (demo_dir / "demo.yaml").exists()
    assert (demo_dir / "config" / "infer.yaml").exists()
    assert (demo_dir / extra_src.name).read_text(encoding="utf-8") == "hello\n"
    assert (demo_dir / "app.py").exists()
    assert (demo_dir / "requirements.txt").exists()
