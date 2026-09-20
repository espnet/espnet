"""Verify native statistics grouping independently of the ESPnet3 adapter."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet2.bin.aggregate_stats_dirs import aggregate_stats_dirs
from espnet2.main_funcs.collect_stats import collect_stats
from espnet2.tasks.asr import ASRTask
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet3.components.data import espnet2_stats
from espnet3.systems.base.training import collect_stats as collect_stage


class FeatureModel(AbsESPnetModel):
    """Expose raw features so grouping differences remain observable."""

    def forward(self, **batch):
        """Provide the model interface required by the native collector."""
        raise NotImplementedError

    def collect_feats(self, speech, speech_lengths, **batch):
        """Keep a scalar feature dimension and its true lengths."""
        return {"feats": speech.unsqueeze(-1), "feats_lengths": speech_lengths}


class PreparedDataset:
    """Represent the processed ESPnet3 side of independent npy inputs."""

    use_espnet_preprocessor = True
    use_espnet_collator = False

    def __init__(self, arrays):
        """Start with training preprocessing enabled to test its suppression."""
        self.arrays = arrays
        self.preprocessor = CommonPreprocessor(train=True)
        self.transforms = [(None, self.preprocessor)]

    def __len__(self):
        """Return the sample count."""
        return len(self.arrays)

    def __getitem__(self, index):
        """Ensure statistics disable augmentation and request ESPnet collation."""
        assert self.use_espnet_collator
        assert not self.preprocessor.train
        return str(index), {"speech": self.arrays[index], "text": np.array([2, 2])}


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_native_statistics_order(tmp_path, monkeypatch, workers):
    """Compare the adapter to original streaming iterators and collector."""
    arrays = [
        np.full(3 + i % 3, [1e5, -1e5, 0.01][i % 3], np.float32) for i in range(11)
    ]
    speech = tmp_path / "speech.scp"
    text = tmp_path / "text"
    rows = []
    for index, array in enumerate(arrays):
        path = tmp_path / f"{index}.npy"
        np.save(path, array)
        rows.append(f"{index} {path}\n")
    speech.write_text("".join(rows))
    text.write_text("".join(f"{i} aa\n" for i in range(len(arrays))))
    dataset = PreparedDataset(arrays)
    organizer = SimpleNamespace(train=dataset, valid=dataset)
    monkeypatch.setattr(espnet2_stats, "instantiate", lambda _: organizer)
    monkeypatch.setattr(espnet2_stats, "get_espnet_model", lambda *args: FeatureModel())
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "model"),
            "stats_dir": str(tmp_path / "new"),
            "task": "fixture",
            "dataset": {},
            "seed": 0,
            "model": {
                "normalize": "global_mvn",
                "normalize_conf": {"stats_file": "keep"},
            },
            "espnet2_stats": {"nj": 3, "batch_size": 3, "num_workers": workers},
        }
    )
    collect_stage(config)
    assert config.model.normalize_conf.stats_file == "keep"
    paths = []
    # split_scp.pl assigns the 11 utterances to contiguous lengths 4, 4, 3.
    for shard, indices in enumerate([range(4), range(4, 8), range(8, 11)]):
        key_file = tmp_path / f"keys.{shard}"
        key_file.write_text("".join(rows[i] for i in indices))
        loader = ASRTask.build_streaming_iterator(
            [(str(speech), "speech", "npy"), (str(text), "text", "text")],
            preprocess_fn=CommonPreprocessor(
                train=False, token_type="char", token_list=["<blank>", "<unk>", "a"]
            ),
            collate_fn=CommonCollateFn(int_pad_value=-1),
            key_file=str(key_file),
            batch_size=3,
            num_workers=workers,
            dtype="float32",
        )
        path = tmp_path / f"native.{shard}"
        collect_stats(FeatureModel(), loader, loader, path, 0, 100, False)
        paths.append(path)
    aggregate_stats_dirs(paths, tmp_path / "native", "WARNING", False)
    for split in ("train", "valid"):
        old = np.load(tmp_path / "native" / split / "feats_stats.npz")
        new = np.load(tmp_path / "new" / split / "feats_stats.npz")
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
        for key in ("speech", "text"):
            assert (tmp_path / "native" / split / f"{key}_shape").read_text() == (
                tmp_path / "new" / split / f"{key}_shape"
            ).read_text()
