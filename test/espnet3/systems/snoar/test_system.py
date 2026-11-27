import pytest
import torch
from omegaconf import OmegaConf

from espnet3.systems.snoar.src.data.manifest import (
    describe_recordings,
    list_audio,
    read_manifest,
    write_manifest,
)
from espnet3.systems.snoar.src.model import SNOARModel
from espnet3.systems.snoar.system import SNOARSystem, resolve_last_checkpoint

from .conftest import tiny_net


def test_stage_order():
    assert [s.name for s in SNOARSystem.stages] == [
        "create_dataset",
        "harvest_noise",
        "train",
        "export",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
    ]


def test_harvest_noise_stage(wav_dir, tmp_path):
    manifest = tmp_path / "train.jsonl"
    write_manifest(describe_recordings(list_audio(wav_dir)), manifest)
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "harvest_noise": {
                "manifest": str(manifest),
                "vad": "energy",
                "out": str(tmp_path / "pool.jsonl"),
            },
        }
    )
    SNOARSystem(training_config=cfg).harvest_noise()
    assert read_manifest(tmp_path / "pool.jsonl")[0]["group"] == "rec0"


def test_export_stage_writes_the_teacher(tmp_path):
    model_cfg = {
        "_target_": "espnet3.systems.snoar.src.model.SNOARModel",
        "mode": "dnf",
        "net": tiny_net(2),
    }
    model = SNOARModel("dnf", net=tiny_net(2))
    ckpt = tmp_path / "last.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path),
            "model": model_cfg,
            "export": {"checkpoint": str(ckpt), "out_dir": str(tmp_path / "out")},
        }
    )
    out = SNOARSystem(training_config=cfg).export()
    meta = OmegaConf.load(out / "snoar.yaml")
    assert meta.readout == "dnf" and meta.mode == "dnf"
    assert (out / "model.pth").is_file() and (out / "config.yaml").is_file()


def test_export_after_resume_takes_the_newest_last_checkpoint(tmp_path):
    """A resumed run leaves last.ckpt at the resume point and writes last-v1."""
    import os

    old, new = SNOARModel("dnf", net=tiny_net(2)), SNOARModel("dnf", net=tiny_net(2))
    old.num_updates.fill_(10)
    new.num_updates.fill_(100)
    torch.save({"state_dict": old.state_dict()}, tmp_path / "step10.ckpt")
    torch.save({"state_dict": new.state_dict()}, tmp_path / "step100.ckpt")
    (tmp_path / "last.ckpt").symlink_to("step10.ckpt")
    (tmp_path / "last-v1.ckpt").symlink_to("step100.ckpt")
    os.utime(tmp_path / "step10.ckpt", (1, 1))
    cfg = OmegaConf.create(
        {
            "exp_dir": str(tmp_path),
            "model": {
                "_target_": "espnet3.systems.snoar.src.model.SNOARModel",
                "mode": "dnf",
                "net": tiny_net(2),
            },
            "export": {
                "checkpoint": str(tmp_path / "last.ckpt"),
                "out_dir": str(tmp_path / "out"),
            },
        }
    )
    out = SNOARSystem(training_config=cfg).export()
    meta = OmegaConf.load(out / "snoar.yaml")
    assert meta.num_updates == 100 and meta.checkpoint.endswith("last-v1.ckpt")
    assert resolve_last_checkpoint(tmp_path / "step10.ckpt") == tmp_path / "step10.ckpt"


def test_stage_errors(tmp_path):
    system = SNOARSystem(training_config=OmegaConf.create({"exp_dir": str(tmp_path)}))
    with pytest.raises(RuntimeError, match="harvest_noise must be set"):
        system.harvest_noise()
    system = SNOARSystem(
        training_config=OmegaConf.create(
            {
                "exp_dir": str(tmp_path),
                "export": {"checkpoint": str(tmp_path / "no.ckpt")},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="run `train` first"):
        system.export()
    with pytest.raises(TypeError, match="does not accept arguments"):
        system.export("x")
