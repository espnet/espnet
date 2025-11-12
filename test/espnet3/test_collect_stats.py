# tests/test_collect_stats.py
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Import the functions under test (adjust import path if your file/module path differs)
from espnet3.collect_stats import collect_stats, collect_stats_multiple_iterator

mp.set_start_method("fork", force=True)

# ===============================================================
# Test Case Summary for Collect Stats
# ===============================================================

# Normal Cases
# | Test Function Name                  | Description                          |
# |--------------------------------------|-------------------------------------|
# | test_collect_stats_local_basic       | Verifies that local mode aggregates |
# | test_collect_stats_local_basic       |  counts/sums correctly and       　 |
# |                                      | writes expected feature/statistics files    |
# | test_collect_stats_parallel_basic    | Verifies that parallel mode         |
# |                                      | (setup_fn + parallel_for) matches local     |
# |                                      | aggregation logic and writes expected files |
# | test_collect_stats_multiple_iterator | Verifies multiple-iterator (sharded) mode   |
# |                                      | aggregates counts, writes stats,            |
# |                                      | and outputs shard-specific shape files      |
# | test_collect_stats_entry_point_basic | Verifies top-level collect_stats() function |
# |                                      | works for different modes and produces      |
# |                                      | correct aggregated results                  |


# -----------------------------
# Dummy components for testing
# -----------------------------


class DummyDataset:
    """Minimal dataset that returns (uid, sample_dict).

    Each item has a variable time length and fixed feature dim.
    """

    def __init__(self, n=10, base_len=3, dim=4, use_espnet_preprocessor=False):
        self.n = n
        self.base_len = base_len
        self.dim = dim
        # Precompute lengths to keep things deterministic
        self.lengths = [self.base_len + (i % 3) for i in range(self.n)]
        self.use_espnet_preprocessor = use_espnet_preprocessor

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        uid = f"utt{idx:03d}"
        T = self.lengths[idx]
        # Create a simple pattern to let us verify sums deterministically
        # Shape: [T, D]
        x = torch.full((T, self.dim), float(idx), dtype=torch.float32)
        if self.use_espnet_preprocessor:
            return uid, {"x": x, "length": T}
        else:
            return {"x": x, "length": T}

    # For multiple-iterator tests: split items into shards round-robin
    def shard(self, shard_idx: int, num_shards: int = 2):
        shard = DummyDataset(n=0, base_len=self.base_len, dim=self.dim)
        indices = [i for i in range(self.n) if i % num_shards == shard_idx]
        shard.n = len(indices)
        shard.lengths = [self.lengths[i] for i in indices]
        # wrap original __getitem__ to map local shard index to global index
        shard._global_indices = indices

        def _getitem(local_idx):
            global_idx = shard._global_indices[local_idx]
            uid = f"utt{global_idx:03d}"
            T = self.lengths[global_idx]
            x = torch.full((T, self.dim), float(global_idx), dtype=torch.float32)
            return uid, {"x": x, "length": T}

        shard.__getitem__ = _getitem  # type: ignore
        return shard


class DummyOrganizer:
    """Hydra-instantiable organizer that exposes .train and .valid datasets."""

    def __init__(
        self, n_train=8, n_valid=5, base_len=3, dim=4, use_espnet_preprocessor=False
    ):
        self.train = DummyDataset(
            n=n_train,
            base_len=base_len,
            dim=dim,
            use_espnet_preprocessor=use_espnet_preprocessor,
        )
        self.valid = DummyDataset(
            n=n_valid,
            base_len=base_len,
            dim=dim,
            use_espnet_preprocessor=use_espnet_preprocessor,
        )


class DummyCollate:
    """Collate that pads to max length and returns:

    (uids, {"x": [B, T, D], "lengths": [B]})
    """

    def __init__(self, int_pad_value: int = -1):
        self.pad = int_pad_value

    def __call__(self, items):
        uids = [u for (u, _) in items]
        seqs = [s["x"] for (_, s) in items]
        lengths = torch.tensor([s["length"] for (_, s) in items], dtype=torch.long)
        max_len = int(max([len(x) for x in seqs]))
        dim = int(seqs[0].shape[-1])
        B = len(seqs)
        x = torch.full((B, max_len, dim), self.pad, dtype=torch.float32)
        for i, seq in enumerate(seqs):
            T = seq.shape[0]
            x[i, :T] = seq
        return uids, {"x": x, "lengths": lengths}


class DummyModel:
    """Model with collect_feats(**batch) that returns torch tensors:

    - "mel": [B, T, D] (copied from input)
    - "mel_lengths": [B]
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def collect_feats(self, *, x: torch.Tensor, lengths: torch.Tensor):
        # Return the batch as "mel", scaled to let us test aggregation
        mel = (x * self.scale).to(self.device)
        mel_lengths = lengths.to(self.device)
        return {"mel": mel, "mel_lengths": mel_lengths}


# ---------------------------------------
# Hydra configs for instantiate(...) calls
# ---------------------------------------


def make_model_cfg(scale: float = 1.0):
    # Use a direct class reference to avoid import path brittleness
    return OmegaConf.create(
        {"_target_": "test.espnet3.test_collect_stats.DummyModel", "scale": scale}
    )


def make_dataset_cfg(
    n_train=8, n_valid=5, base_len=3, dim=4, use_espnet_preprocessor=False
):
    return OmegaConf.create(
        {
            "_target_": "test.espnet3.test_collect_stats.DummyOrganizer",
            "n_train": n_train,
            "n_valid": n_valid,
            "base_len": base_len,
            "dim": dim,
            "use_espnet_preprocessor": use_espnet_preprocessor,
        }
    )


def make_dataloader_cfg(
    use_custom_collate: bool = True,
    multiple_iterator: bool = False,
    num_shards: int = 2,
):
    if use_custom_collate:
        return OmegaConf.create(
            {
                "train": {
                    "multiple_iterator": multiple_iterator,
                    "num_shards": num_shards,
                },
                "valid": {
                    "multiple_iterator": multiple_iterator,
                    "num_shards": num_shards,
                },
                "collate_fn": {
                    "_target_": "test.espnet3.test_collect_stats.DummyCollate",
                    "int_pad_value": -1,
                },
            }
        )
    else:
        # Fallback path to CommonCollateFn (not used here)
        return OmegaConf.create(
            {
                "train": {
                    "multiple_iterator": multiple_iterator,
                    "num_shards": num_shards,
                },
                "valid": {
                    "multiple_iterator": multiple_iterator,
                    "num_shards": num_shards,
                },
            }
        )


def make_parallel_cfg(n_workers=2):
    # Matches your parallel.py expectations
    return OmegaConf.create({"env": "local", "n_workers": n_workers, "options": {}})


# -----------------
# Helper assertions
# -----------------


def _expected_total_count(dataset):
    # Sum of all per-utterance time lengths
    return sum(dataset.lengths)


def _load_npz_counts(dirpath: Path, feat_key: str):
    p = dirpath / f"{feat_key}_stats.npz"
    assert p.exists(), f"Expected stats file not found: {p}"
    data = np.load(p)
    return data["count"], data["sum"], data["sum_square"]


# ------------
# The tests
# ------------
@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("use_parallel", [False, True])
@pytest.mark.parametrize("use_espnet_preprocessor", [False, True])
def test_collect_stats_local_basic(
    tmp_path: Path, use_parallel, use_espnet_preprocessor
):
    # Verify that local (non-parallel) path aggregates counts/sums correctly
    # and writes expected files.
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_dataset_cfg(
        n_train=6,
        n_valid=0,
        base_len=3,
        dim=4,
        use_espnet_preprocessor=use_espnet_preprocessor,
    )
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    par_cfg = make_parallel_cfg(n_workers=2) if use_parallel else None

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run "train" split locally
    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=par_cfg,
        write_collected_feats=True,
        batch_size=3,
    )

    mode_dir = out_dir / "train"
    assert (mode_dir / "stats_keys").exists(), "stats_keys not written"

    for k in ["mel", "mel_lengths"]:
        npz = mode_dir / f"{k}_stats.npz"
        assert npz.exists(), f"{npz} missing"
        scp = mode_dir / "collect_feats" / f"{k}.scp"
        assert scp.exists(), f"{scp} missing"

    ds = instantiate(ds_cfg).train
    total_count = _expected_total_count(ds)
    cnt, s, sq = _load_npz_counts(mode_dir, "mel")
    assert int(cnt) == total_count, "Total frame count mismatch for mel"


@pytest.mark.execution_timeout(30)
def test_collect_stats_multiple_iterator(tmp_path: Path):
    # Verify multiple-iterator (sharded) path aggregates counts and writes shapes
    # with shard suffixes. Feature saving is disabled by spec.
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_dataset_cfg(n_train=10, n_valid=0, base_len=2, dim=2)
    # multiple_iterator=True and define number of shards
    dl_cfg = make_dataloader_cfg(
        use_custom_collate=True, multiple_iterator=True, num_shards=2
    )
    par_cfg = make_parallel_cfg(n_workers=2)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_dict, sq_dict, count_dict = collect_stats_multiple_iterator(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        batch_size=2,
        parallel_config=par_cfg,
    )

    # Persist like collect_stats does
    mode_dir = out_dir / "train"
    for k in sum_dict:
        np.savez(
            mode_dir / f"{k}_stats.npz",
            count=count_dict[k],
            sum=sum_dict[k],
            sum_square=sq_dict[k],
        )
    with open(mode_dir / "stats_keys", "w") as f:
        f.write("\n".join(sum_dict) + "\n")

    # Assertions
    ds = instantiate(ds_cfg).train
    total_count = _expected_total_count(ds)
    cnt, s, sq = _load_npz_counts(mode_dir, "mel")
    assert (
        int(cnt) == total_count
    ), "Total frame count mismatch in multiple-iterator mode"

    # Shape files with shard suffix should exist (at least for one key)
    # We don't know exact uids, but file creation per shard is enough signal.
    # Look for any file starting with "mel_shape.shard."
    shard_shape_files = (
        list((mode_dir / "db").glob("mel_shape.shard.*"))
        if (mode_dir / "db").exists()
        else []
    )
    # If your DatadirWriter writes into mode_dir directly (not "db"), adjust:
    if not shard_shape_files:
        shard_shape_files = list(mode_dir.glob("mel_shape.shard.*"))
    assert shard_shape_files != [], "No shard shape files found"


# ----------------------------
# Entry-point level smoke tests
# ----------------------------


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("use_parallel", [False, True])
def test_collect_stats_entrypoint_train(tmp_path: Path, use_parallel):
    # Smoke-test the public entrypoint `collect_stats` for the 'train' split,
    # both in local (no parallel_config) and parallel (with parallel_config) modes.
    # erifies that stats are saved and key files exist.
    # Also checks that *_lengths features are persisted (since they matter).
    model_cfg = make_model_cfg(scale=1.5)
    ds_cfg = make_dataset_cfg(n_train=6, n_valid=0, base_len=3, dim=4)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    par_cfg = make_parallel_cfg(n_workers=2) if use_parallel else None

    out_dir = tmp_path / "out_ep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Call the public entrypoint
    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=par_cfg,
        write_collected_feats=True,
        batch_size=2,
    )

    mode_dir = out_dir / "train"

    # Basic files
    assert (mode_dir / "stats_keys").exists(), "stats_keys not written by entrypoint"
    # Stats exist for both mel and mel_lengths
    for k in ["mel", "mel_lengths"]:
        npz = mode_dir / f"{k}_stats.npz"
        assert npz.exists(), f"{npz} not written by entrypoint"

    # Feature scps exist (including lengths)
    for k in ["mel", "mel_lengths"]:
        scp = mode_dir / "collect_feats" / f"{k}.scp"
        assert scp.exists(), f"{scp} not written by entrypoint"

    # Count check matches dataset total frames for mel
    ds = instantiate(ds_cfg).train
    total_count = _expected_total_count(ds)
    cnt, s, sq = _load_npz_counts(mode_dir, "mel")
    assert int(cnt) == total_count, "Total frame count mismatch in entrypoint(train)"


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("use_parallel", [False, True])
def test_collect_stats_entrypoint_valid(tmp_path: Path, use_parallel):
    # Same as above but for 'valid' split, to ensure both branches work.
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_dataset_cfg(n_train=0, n_valid=5, base_len=2, dim=3)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    par_cfg = make_parallel_cfg(n_workers=2) if use_parallel else None

    out_dir = tmp_path / "out_ep_valid"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="valid",
        output_dir=out_dir,
        task=None,
        parallel_config=par_cfg,
        write_collected_feats=True,
        batch_size=2,
    )

    mode_dir = out_dir / "valid"
    assert (mode_dir / "stats_keys").exists()
    for k in ["mel", "mel_lengths"]:
        assert (mode_dir / f"{k}_stats.npz").exists()
        assert (mode_dir / "collect_feats" / f"{k}.scp").exists()


@pytest.mark.execution_timeout(30)
def test_collect_stats_entrypoint_multiple_iterator(tmp_path: Path):
    # Entrypoint with multiple_iterator=True should dispatch to the sharded path.
    # It must save stats and shard-suffixed shapes; features are not saved by spec.
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_dataset_cfg(n_train=10, n_valid=0, base_len=2, dim=2)
    dl_cfg = make_dataloader_cfg(
        use_custom_collate=True, multiple_iterator=True, num_shards=2
    )
    par_cfg = make_parallel_cfg(n_workers=2)

    out_dir = tmp_path / "out_ep_multi"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=par_cfg,
        write_collected_feats=False,  # enforced by collect_stats
        batch_size=2,
    )

    mode_dir = out_dir / "train"
    assert (mode_dir / "stats_keys").exists()

    # Stats files exist
    for k in ["mel", "mel_lengths"]:
        assert (mode_dir / f"{k}_stats.npz").exists()

    # No feature scps in multi-iterator mode
    cf_dir = mode_dir / "collect_feats"
    assert not cf_dir.exists() or list(cf_dir.glob("*.scp")) == []

    # Shard shape files exist
    shard_shape_files = list(mode_dir.glob("mel_shape.shard.*"))
    assert (
        shard_shape_files
    ), "Shard shape files were not written in entrypoint(multiple_iterator)"
