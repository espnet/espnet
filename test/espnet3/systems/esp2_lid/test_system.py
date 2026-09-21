from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

import espnet3.systems.esp2_lid.collect_stats as stats_module
from espnet3.systems.esp2_lid.system import LIDSystem
from espnet3.utils.config_utils import load_and_merge_config


class DummyDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return {
            "speech": np.zeros(self.lengths[index], dtype=np.float32),
            "lid_labels": "eng",
        }


def test_collect_speech_shapes(tmp_path, monkeypatch):
    organizer = SimpleNamespace(
        train=DummyDataset([3, 5]),
        valid=DummyDataset([4]),
    )
    seen = {}

    def fake_instantiate(config):
        seen["preprocessor"] = config.preprocessor
        return organizer

    monkeypatch.setattr(stats_module, "instantiate", fake_instantiate)
    config = OmegaConf.create(
        {
            "stats_dir": str(tmp_path),
            "dataset": {"preprocessor": {"_target_": "unused"}},
            "dataloader": {
                "train": {"iter_factory": {"num_workers": 0}},
                "valid": {"iter_factory": {"num_workers": 0}},
            },
        }
    )

    stats_module.collect_speech_shapes(config)

    assert seen["preprocessor"] is None
    assert config.dataset.preprocessor._target_ == "unused"
    assert (tmp_path / "train/speech_shape").read_text() == "0 3\n1 5\n"
    assert (tmp_path / "valid/speech_shape").read_text() == "0 4\n"
    for mode in ("train", "valid"):
        assert (tmp_path / mode / "stats_keys").read_text() == "\n"
        assert (tmp_path / mode / "batch_keys").read_text() == "speech\n"
        assert not (tmp_path / mode / "lid_labels_shape").exists()
        assert not list((tmp_path / mode).glob("*_stats.npz"))


@pytest.mark.parametrize("extract_features", [False, True, None])
def test_lid_system_collect_stats(tmp_path, extract_features):
    """Bootstrap metadata and collect unpadded features from a fresh recipe."""
    recipe = Path(__file__).resolve().parents[4] / "egs3/voxlingua107/esp2_lid"
    config = load_and_merge_config(
        recipe / "conf/training.yaml", "training.yaml", resolve=False
    )
    config.recipe_dir = str(tmp_path)
    config.model = {
        "lang_num": 2,
        "frontend": "default",
        "frontend_conf": {"n_fft": 64, "hop_length": 32, "n_mels": 8},
        "encoder": "identity",
        "encoder_conf": {},
        "pooling": "mean",
        "pooling_conf": {},
        "projector": "xvector",
        "projector_conf": {"output_size": 4},
        "loss": "softmax",
        "loss_conf": {},
        "model_conf": (
            {}
            if extract_features is None
            else {"extract_feats_in_collect_stats": extract_features}
        ),
    }
    config.trainer.accelerator = "cpu"
    config.trainer.precision = "32-true"
    lengths = [640, 800, 960, 1120, 1280]
    for mode, split in (("train", "train"), ("valid", "dev")):
        config.dataset[mode][0].data_src = "egs3.voxlingua107.esp2_lid.dataset"
        config.dataloader[mode].iter_factory.num_workers = 0
        directory = tmp_path / "data/voxlingua107" / split
        directory.mkdir(parents=True)
        rows = []
        for index, length in enumerate(lengths):
            audio = directory / f"{index}.wav"
            sf.write(audio, 0.1 * np.sin(np.arange(length) * 0.1), 16000)
            language = "eng" if index % 2 == 0 else "fra"
            rows.append(f"{index}\t{audio}\t{language}\n")
        (directory / "manifest.tsv").write_text("".join(rows))

    LIDSystem(training_config=config).collect_stats()

    for mode in ("train", "valid"):
        directory = Path(config.stats_dir) / mode
        for name in ("lang2utt", "category2utt"):
            assert (directory / name).read_text() == "eng 0 2 4\nfra 1 3\n"
        assert (directory / "speech_shape").read_text() == "".join(
            f"{index} {length}\n" for index, length in enumerate(lengths)
        )
        assert (directory / "batch_keys").read_text() == "speech\n"
        if extract_features is False:
            assert (directory / "stats_keys").read_text() == "\n"
            assert not (directory / "feats_stats.npz").exists()
        else:
            frames = [length // 32 + 1 for length in lengths]
            assert (directory / "feats_shape").read_text() == "".join(
                f"{index} {count},8\n" for index, count in enumerate(frames)
            )
            assert "feats" in (directory / "stats_keys").read_text().splitlines()
            with np.load(directory / "feats_stats.npz") as stats:
                assert stats["count"] == sum(frames)
                assert stats["sum"].shape == (8,)
                assert np.isfinite(stats["sum_square"]).all()
