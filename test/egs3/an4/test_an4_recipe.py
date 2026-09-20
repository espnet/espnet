"""Regression tests for full AN4 preparation and migration settings."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from hydra.utils import instantiate

from egs3.an4.asr.dataset import Dataset, DatasetBuilder
from egs3.an4.asr.dataset.builder import read_manifest
from egs3.an4.asr.src.inference import build_output
from egs3.an4.asr.src.tokenizer import gather_training_text
from espnet3.utils.config_utils import load_and_merge_config

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "egs3/an4/asr"


@pytest.fixture
def corpus(tmp_path):
    """Create unsorted transcripts and real NIST audio, without a download."""
    source = tmp_path / "downloads/an4"
    (source / "etc").mkdir(parents=True)
    for split, speakers in (("train", ("zz", "bb", "aa")), ("test", ("tt",))):
        subdir = "an4_clstk" if split == "train" else "an4test_clstk"
        lines = []
        for speaker in speakers:
            recording = f"an1-{speaker}-b"
            path = source / "wav" / subdir / speaker / f"{recording}.sph"
            path.parent.mkdir(parents=True, exist_ok=True)
            signal = (1000 * np.sin(np.arange(16000) * 0.1)).astype(np.int16)
            sf.write(path, signal, 16000, format="NIST", subtype="PCM_16")
            lines.append(f"<s> TEXT {speaker.upper()} </s> ({recording})\n")
        (source / f"etc/an4_{split}.transcription").write_text("".join(lines))
    return tmp_path


def test_checks_are_read_only(tmp_path):
    """Preparation checks must leave a fresh directory untouched."""
    builder = DatasetBuilder()
    assert not builder.is_source_prepared(recipe_dir=tmp_path)
    assert not builder.is_built(recipe_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_split_augmentation_and_pcm(corpus):
    """Sort and split before perturbation, preserving unmodified PCM."""
    builder = DatasetBuilder()
    builder.build(recipe_dir=corpus, dev_size=1)
    assert builder.is_built(recipe_dir=corpus, dev_size=1)
    assert not builder.is_built(recipe_dir=corpus, dev_size=2)
    train = Dataset("train", recipe_dir=corpus)
    valid = Dataset("valid", recipe_dir=corpus)
    test = Dataset("test", recipe_dir=corpus)
    assert (len(train), len(valid), len(test)) == (6, 1, 1)
    assert valid.entries[0][0] == "aa-an1-b"
    assert {entry[2] for entry in train.entries} == {"TEXT BB", "TEXT ZZ"}
    assert {entry[2] for entry in test.entries} == {"TEXT TT"}
    assert [entry[0] for entry in train.entries] == [
        "bb-an1-b",
        "sp0.9-bb-an1-b",
        "sp0.9-zz-an1-b",
        "sp1.1-bb-an1-b",
        "sp1.1-zz-an1-b",
        "zz-an1-b",
    ]
    assert gather_training_text(corpus / "data/manifest/train.tsv") == [
        row[2] for row in train.entries
    ]
    for index, (uid, _, _) in enumerate(train.entries):
        sample = train[index]
        assert set(sample) == {"speech", "text"}
        assert sample["speech"].dtype == np.float32
        speed = float(uid.split("-")[0][2:]) if uid.startswith("sp") else 1.0
        assert abs(len(sample["speech"]) - 16000 / speed) <= 1
    source = corpus / "downloads/an4/wav/an4_clstk/aa/an1-aa-b.sph"
    np.testing.assert_array_equal(
        valid[0]["speech"], sf.read(source, dtype="float32")[0]
    )


@pytest.mark.parametrize("empty_speaker", ["aa", "bb"])
def test_empty_transcripts_follow_espnet2_filtering(corpus, empty_speaker):
    """Filter empty valid text after splitting, but preserve test text."""
    source = corpus / "downloads/an4/etc"
    train_path = source / "an4_train.transcription"
    train_path.write_text(
        train_path.read_text().replace(f"TEXT {empty_speaker.upper()}", "")
    )
    test_path = source / "an4_test.transcription"
    test_path.write_text(test_path.read_text().replace("TEXT TT", ""))
    DatasetBuilder().build(recipe_dir=corpus, dev_size=2)
    valid = Dataset("valid", recipe_dir=corpus)
    train = Dataset("train", recipe_dir=corpus)
    test = Dataset("test", recipe_dir=corpus)
    assert len(valid) == 1
    assert len(train) == 3
    assert all(entry[2].strip() for entry in valid.entries + train.entries)
    assert len(test) == 1 and test.entries[0][2] == ""


def test_empty_training_transcript_is_filtered(corpus):
    """Remove all speed variants of an empty training transcript."""
    path = corpus / "downloads/an4/etc/an4_train.transcription"
    path.write_text(path.read_text().replace("TEXT BB", ""))
    DatasetBuilder().build(recipe_dir=corpus, dev_size=1)
    train = Dataset("train", recipe_dir=corpus)
    assert len(train) == 3
    assert all(entry[2] == "TEXT ZZ" for entry in train.entries)


def test_reject_incomplete_source_and_invalid_split(corpus):
    """Fail before publishing invalid or incomplete data."""
    builder = DatasetBuilder()
    with pytest.raises(ValueError, match="dev_size"):
        builder.build(recipe_dir=corpus, dev_size=3)
    assert not builder.is_built(recipe_dir=corpus)
    with pytest.raises(ValueError, match="Unknown AN4 split"):
        Dataset("unrecognized", recipe_dir=corpus)
    with pytest.raises(FileNotFoundError, match="Incomplete AN4"):
        builder.prepare_source(source_dir=corpus / "missing")


def test_source_model_and_optimizer_equivalence():
    """Instantiate the merged config and compare it to the source recipe."""
    import torch

    source = yaml.safe_load(
        (ROOT / "egs2/an4/asr1/conf/train_asr_sinc_rnn.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_sinc_rnn.yaml", "training.yaml"
    )
    for key in (
        "init",
        "frontend",
        "frontend_conf",
        "preencoder",
        "encoder",
        "encoder_conf",
        "decoder",
        "decoder_conf",
    ):
        assert config.model[key] == source[key]
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = instantiate(config.optimizer, [parameter])
    scheduler = instantiate(config.scheduler, optimizer=optimizer)
    assert isinstance(optimizer, torch.optim.Adadelta)
    for key, value in source["optim_conf"].items():
        assert optimizer.param_groups[0][key] == value
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert "warmup_steps" not in config.scheduler
    for key, value in source["scheduler_conf"].items():
        assert getattr(scheduler, key) == value
    assert config.trainer.max_epochs == source["max_epoch"]
    assert config.trainer.callbacks[0].patience == source["patience"] + 1
    assert config.dataset._recursive_ is False
    assert config.dataloader.train.iter_factory.batches.fold_lengths == [80000, 150]


def test_output_pairs_single_and_batched_references():
    """Keep hypotheses aligned with their references in both modes."""
    data = {"text": "HELLO"}
    hypothesis = [("WORLD", None, None, None)]
    assert build_output(data, hypothesis, 3) == {
        "utt_id": "3",
        "ref": "HELLO",
        "hyp": "WORLD",
    }
    outputs = build_output([data, data], [hypothesis, [(None,)]], [3, 4])
    assert outputs[1] == {"utt_id": "4", "ref": "HELLO", "hyp": ""}
    with pytest.raises(ValueError):
        build_output([data, data], [hypothesis], [3, 4])


def test_manifest_rejects_duplicate_ids(tmp_path):
    """Reject ambiguous utterance identifiers."""
    path = tmp_path / "duplicate.tsv"
    path.write_text("same\t/a.wav\tA\nsame\t/b.wav\tB\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path)


@pytest.mark.parametrize("fail", [False, True])
def test_statistics_preserve_normalization_and_write_input_shapes(
    tmp_path, monkeypatch, fail
):
    """Protect subsequent training and retain speech/text fold semantics."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from egs3.an4.asr.src import system as module

    config = OmegaConf.create(
        {
            "model": {
                "normalize": "global_mvn",
                "normalize_conf": {"stats_file": "stats"},
            },
            "stats_dir": str(tmp_path),
            "dataset": {},
        }
    )
    system = object.__new__(module.An4System)
    system.training_config = config
    monkeypatch.setattr(system, "train_tokenizer", lambda: None)

    def collect(self):
        self.training_config.model.pop("normalize")
        self.training_config.model.pop("normalize_conf")
        if fail:
            raise RuntimeError("Interrupted statistics")

    monkeypatch.setattr(module.ASRSystem, "collect_stats", collect)
    samples = [{"speech": np.zeros(16000), "text": np.arange(7)}]
    monkeypatch.setattr(
        module, "instantiate", lambda _: SimpleNamespace(train=samples, valid=samples)
    )
    if fail:
        with pytest.raises(RuntimeError, match="Interrupted statistics"):
            system.collect_stats()
    else:
        system.collect_stats()
        assert (tmp_path / "train/speech_shape").read_text() == "0 16000\n"
        assert (tmp_path / "valid/text_shape").read_text() == "0 7\n"
    assert system.training_config is config
    assert config.model.normalize == "global_mvn"
    assert config.model.normalize_conf.stats_file == "stats"


def test_model_matches_espnet2_initial_state():
    """Build both task models with one seed and compare all initialized tensors."""
    import torch
    from omegaconf import OmegaConf

    from espnet3.utils.task_utils import get_espnet_model

    source = yaml.safe_load(
        (ROOT / "egs2/an4/asr1/conf/train_asr_sinc_rnn.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_sinc_rnn.yaml", "training.yaml"
    )
    target = OmegaConf.to_container(config.model, resolve=True)
    tokens = ["<blank>", "<unk>", *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "▁", "<sos/eos>"]
    for arguments in (source, target):
        arguments.update(token_list=tokens, normalize=None, normalize_conf={})
    torch.manual_seed(0)
    original = get_espnet_model("espnet2.tasks.asr.ASRTask", source)
    torch.manual_seed(0)
    migrated = get_espnet_model("espnet3.systems.asr.task.ASRTask", target)
    assert original.ctc_weight == migrated.ctc_weight == 0.5
    before, after = original.state_dict(), migrated.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_best_checkpoint_selection(tmp_path):
    """Use the retained best model, not a last or stale averaged checkpoint."""
    from egs3.an4.asr.src.inference import find_best_checkpoint

    (tmp_path / "step10.ckpt").touch()
    (tmp_path / "valid.acc.ave_1best.pth").touch()
    with pytest.raises(RuntimeError, match="found 0"):
        find_best_checkpoint(tmp_path)
    best = tmp_path / "epoch0_step2_valid.acc.ckpt"
    best.touch()
    assert find_best_checkpoint(tmp_path) == best
    (tmp_path / "epoch1_step4_valid.acc.ckpt").touch()
    with pytest.raises(RuntimeError, match="found 2"):
        find_best_checkpoint(tmp_path)


def test_test_split_is_not_duration_filtered(corpus):
    """Match ESPnet2 stage 4, which filters train/valid but preserves test audio."""
    path = corpus / "downloads/an4/wav/an4test_clstk/tt/an1-tt-b.sph"
    sf.write(path, np.zeros(800, dtype=np.int16), 16000, format="NIST")
    DatasetBuilder().build(recipe_dir=corpus, dev_size=1)
    assert len(Dataset("test", recipe_dir=corpus)[0]["speech"]) == 800


def test_early_stopping_matches_espnet2_patience():
    """Both trainers must tolerate four bad epochs and stop on the fifth."""
    import torch
    from lightning.pytorch.callbacks import EarlyStopping

    config = load_and_merge_config(
        RECIPE / "conf/training_sinc_rnn.yaml", "training.yaml"
    )
    callback = instantiate(config.trainer.callbacks[0])
    assert isinstance(callback, EarlyStopping)
    stopped = [
        callback._evaluate_stopping_criteria(torch.tensor(loss))[0]
        for loss in [1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    ]
    assert stopped == [False, False, False, False, False, True]


def test_download_extraction_and_lm_text(corpus, tmp_path, monkeypatch):
    """Exercise a local archive through the real extraction and preparation path."""
    import shutil
    import tarfile

    from egs3.an4.asr.dataset import builder as module

    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(corpus / "downloads/an4", arcname="an4")

    def download(url, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, path)

    monkeypatch.setattr(module, "download_url", download)
    output = tmp_path / "extracted"
    builder = DatasetBuilder()
    builder.prepare_source(recipe_dir=output)
    builder.build(recipe_dir=output, dev_size=1)
    assert builder.is_built(recipe_dir=output, dev_size=1)
    assert len((output / "data/lm/train.txt").read_text().splitlines()) == 6
    assert len((output / "data/lm/valid.txt").read_text().splitlines()) == 1
