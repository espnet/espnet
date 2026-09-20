"""VOiCES devkit preparation and ESPnet2 migration regression tests."""

import csv
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egs3.voices.asr.dataset import Dataset, DatasetBuilder
from egs3.voices.asr.dataset import builder as module
from egs3.voices.asr.src.inference import average_checkpoints, build_output
from egs3.voices.asr.src.tokenizer import gather_training_text
from espnet3.utils.config_utils import load_and_merge_config

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "egs3/voices/asr"


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Model clean sources and two recording variants with separate speakers."""
    config = dict(module._BUILDER_CFG)
    config.update(dev_speakers=2, expected_distant_counts={"train": 8, "test": 2})
    monkeypatch.setattr(module, "_BUILDER_CFG", config)
    source = tmp_path / "downloads/VOiCES_devkit"
    references = source / "references/filename_transcripts"
    references.parent.mkdir(parents=True)
    with references.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "filename", "transcript"])
        for split, speakers in (
            ("train", ["0010", "0002", "0001", "0009"]),
            ("test", ["0020"]),
        ):
            for speaker in speakers:
                stem = f"Lab41-SRI-VOiCES-src-sp{speaker}-ch000001-sg0001"
                path = source / "source-16k" / split / speaker / f"{stem}.wav"
                path.parent.mkdir(parents=True)
                waveform = np.linspace(-0.4, 0.4, 32000, dtype=np.float32)
                sf.write(path, waveform, 16000, subtype="PCM_16")
                for microphone in ("01", "05"):
                    distant = (
                        f"Lab41-SRI-VOiCES-rm1-babb-sp{speaker}-ch000001-sg0001-"
                        f"mc{microphone}-stu-clo-dg030"
                    )
                    path = (
                        source
                        / "distant-16k/speech"
                        / split
                        / speaker
                        / f"{distant}.wav"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, waveform, 16000, subtype="PCM_16")
                    writer.writerow([0, str(path.name), f"WORDS FOR SPEAKER {speaker}"])
    return tmp_path, source


def test_checks_are_read_only(tmp_path):
    """Status probes must not create directories or trigger downloads."""
    builder = DatasetBuilder()
    assert not builder.is_source_prepared(recipe_dir=tmp_path)
    assert not builder.is_built(recipe_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_speaker_split_source_grouping_and_waveforms(corpus):
    """Keep every version of one source in its original speaker's partition."""
    recipe, source = corpus
    builder = DatasetBuilder()
    builder.build(recipe_dir=recipe)
    assert builder.is_built(recipe_dir=recipe)
    assert not builder.is_built(recipe_dir=recipe, source_dir=source / "elsewhere")
    train, valid, test = [Dataset(s, recipe_dir=recipe) for s in module.SPLITS]
    assert [len(d) for d in (train, valid, test)] == [6, 6, 3]
    assert {r["speaker"] for r in valid.entries} == {"0001", "0002"}
    groups = [{r["source_id"] for r in d.entries} for d in (train, valid, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert {r["condition"] for r in test.entries} == {"source", "distant"}
    assert len(Dataset("test", recipe_dir=recipe, condition="distant")) == 2
    assert gather_training_text(recipe / "data/manifest/tokenizer_train.txt") == [
        r["text"] for r in train.entries
    ]
    for dataset in (train, valid, test):
        for index, row in enumerate(dataset.entries):
            sample = dataset[index]
            assert set(sample) == {"speech", "text"}
            assert sample["speech"].dtype == np.float32
            np.testing.assert_array_equal(
                sample["speech"], sf.read(row["path"], dtype="float32")[0]
            )
    marker = recipe / "data/manifest/build.json"
    first = marker.read_bytes()
    builder.build(recipe_dir=recipe)
    assert marker.read_bytes() == first


def test_duration_filter_does_not_filter_test(corpus):
    """Match strict ESPnet2 train/dev duration bounds; retain all test audio."""
    recipe, source = corpus
    train = sorted((source / "source-16k/train").rglob("*.wav"))
    test = next((source / "source-16k/test").rglob("*.wav"))
    sf.write(train[0], np.zeros(1600), 16000)
    sf.write(train[-1], np.zeros(480000), 16000)
    sf.write(test, np.zeros(800), 16000)
    DatasetBuilder().build(recipe_dir=recipe)
    assert len(Dataset("train", recipe_dir=recipe)) == 5
    assert len(Dataset("valid", recipe_dir=recipe)) == 5
    assert len(Dataset("test", recipe_dir=recipe)) == 3
    assert (
        json.loads((recipe / "data/manifest/build.json").read_text())[
            "filtered_train_valid"
        ]
        == 2
    )


def test_reject_incomplete_audio_and_conflicting_transcripts(corpus):
    """Incomplete archives and ambiguous references must fail before training."""
    recipe, source = corpus
    path = next((source / "distant-16k/speech/test").rglob("*.wav"))
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="Incomplete devkit test"):
        DatasetBuilder().build(recipe_dir=recipe)
    path.write_bytes(original)
    with (source / "references/filename_transcripts").open("a") as stream:
        stream.write(f"0,{path.name},CONFLICT\n")
    with pytest.raises(ValueError, match="conflicting"):
        DatasetBuilder().build(recipe_dir=recipe)
    assert not DatasetBuilder().is_built(recipe_dir=recipe)


def test_audio_and_dataset_validation(corpus):
    """Reject unsupported audio and dataset selectors instead of resampling."""
    recipe, source = corpus
    path = next((source / "source-16k/train").rglob("*.wav"))
    sf.write(path, np.zeros(8000), 8000)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        DatasetBuilder().build(recipe_dir=recipe)
    for kwargs in ({"split": "dev"}, {"split": "test", "condition": "unknown"}):
        with pytest.raises(ValueError, match="Unknown VOiCES"):
            Dataset(recipe_dir=recipe, **kwargs)
    with pytest.raises(FileNotFoundError, match="Incomplete extracted"):
        DatasetBuilder().prepare_source(recipe_dir=recipe, source_dir=source / "absent")


def test_shared_download_and_extraction(corpus, tmp_path, monkeypatch):
    """Exercise the same shared extractor with a small local devkit archive."""
    recipe, source = corpus
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname="VOiCES_devkit")
    destination = recipe / "other_recipe"
    called = []

    def download(url, path):
        called.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, path)

    monkeypatch.setattr(module, "download_url", download)
    builder = DatasetBuilder()
    builder.prepare_source(recipe_dir=destination)
    builder.build(recipe_dir=destination)
    assert called == [module._BUILDER_CFG["url"]]
    assert builder.is_built(recipe_dir=destination)
    builder.prepare_source(recipe_dir=destination)
    assert len(called) == 1


def test_model_and_training_settings_match_source():
    """Catch dropped Conformer, SpecAugment, optimizer, or trainer settings."""
    source = yaml.safe_load(
        (ROOT / "egs2/voices/asr1/conf/train_asr_conformer.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_conformer.yaml", "training.yaml"
    )
    for key in (
        "encoder",
        "encoder_conf",
        "decoder",
        "decoder_conf",
        "model_conf",
        "frontend_conf",
        "specaug",
        "specaug_conf",
    ):
        assert config.model[key] == source[key]
    optimizer = instantiate(config.optimizer, [torch.nn.Parameter(torch.ones(1))])
    for key, value in source["optim_conf"].items():
        assert optimizer.param_groups[0][key] == value
    scheduler = instantiate(config.scheduler, optimizer=optimizer)
    assert scheduler.warmup_steps == source["scheduler_conf"]["warmup_steps"]
    assert config.trainer.max_epochs == source["max_epoch"]
    assert config.espnet2_compat.accum_grad == source["accum_grad"]
    assert config.trainer.precision == "16-mixed" and source["use_amp"]
    assert config.best_model_criterion == [["valid/acc", 10, "max"]]
    assert config.dataset._recursive_ is False
    assert config.tokenizer.model_type == "unigram"
    assert (
        config.dataloader.train.iter_factory.batches.batch_bins == source["batch_bins"]
    )


@pytest.mark.parametrize("fail", [False, True])
def test_statistics_preserve_normalization_and_numel_shapes(
    tmp_path, monkeypatch, fail
):
    """Keep GlobalMVN and the text vocabulary dimension required by numel."""
    from egs3.voices.asr.src import system as system_module

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blank>\n<unk>\nA\nB\n<sos/eos>\n")
    config = OmegaConf.create(
        {
            "model": {
                "normalize": "global_mvn",
                "normalize_conf": {"stats_file": "stats"},
                "token_list": str(tokens),
            },
            "stats_dir": str(tmp_path),
            "dataset": {},
        }
    )
    system = object.__new__(system_module.VoicesSystem)
    system.training_config = config
    monkeypatch.setattr(system, "train_tokenizer", lambda: None)

    def collect(self):
        self.training_config.model.pop("normalize")
        self.training_config.model.pop("normalize_conf")
        if fail:
            raise RuntimeError("Interrupted statistics")

    monkeypatch.setattr(system_module.ASRSystem, "collect_stats", collect)
    samples = [{"speech": np.zeros(16000), "text": np.arange(7)}]
    monkeypatch.setattr(
        system_module,
        "instantiate",
        lambda _: SimpleNamespace(train=samples, valid=samples),
    )
    if fail:
        with pytest.raises(RuntimeError, match="Interrupted statistics"):
            system.collect_stats()
    else:
        system.collect_stats()
        assert (tmp_path / "train/speech_shape").read_text() == "0 16000\n"
        assert (tmp_path / "valid/text_shape").read_text() == "0 7,5\n"
    assert system.training_config is config
    assert config.model.normalize == "global_mvn"
    assert config.model.normalize_conf.stats_file == "stats"


@pytest.mark.parametrize("count", [2, 10])
def test_final_average_matches_native(tmp_path, count):
    """Match native short-run fallback and metric-ranked float summation."""
    from espnet2.main_funcs.average_nbest_models import average_nbest_models
    from espnet2.train.reporter import Reporter

    reporter = Reporter()
    scores = {}
    for epoch in range(count):
        value = ([1e20, -1e20, 3.0] + [0.0] * 10)[epoch]
        score = float({0: 10, 2: 9, 1: 8}.get(epoch, 0))
        state = {"weight": torch.tensor([value]), "count": torch.tensor(epoch + 1)}
        torch.save(state, tmp_path / f"{epoch + 1}epoch.pth")
        path = tmp_path / f"epoch{epoch}_step{epoch + 1}_valid.acc.ckpt"
        torch.save({"state_dict": state}, path)
        scores[str(path)] = torch.tensor(score)
        reporter.stats[epoch + 1] = {"valid": {"acc": score}}
    torch.save(
        {"callbacks": {"scores": {"monitor": "valid/acc", "best_k_models": scores}}},
        tmp_path / f"step{count}.ckpt",
    )
    reporter.set_epoch(count)
    average_nbest_models(tmp_path, reporter, [["valid", "acc", "max"]], 10)
    expected = torch.load(tmp_path / "valid.acc.ave.pth", weights_only=True)
    actual = torch.load(average_checkpoints(tmp_path), weights_only=True)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    with pytest.raises(RuntimeError, match="Expected"):
        average_checkpoints(tmp_path, max_checkpoints=1)
    with pytest.raises(RuntimeError, match="Expected"):
        average_checkpoints(tmp_path / "absent")
    (tmp_path / f"step{count}.ckpt").unlink()
    with pytest.raises(RuntimeError, match="Missing final"):
        average_checkpoints(tmp_path)


def test_output_alignment_and_empty_prediction():
    """Keep every reference, including when a smoke model predicts no text."""
    output = build_output([{"text": "A"}, {"text": "B"}], [[("C",)], [(None,)]], [3, 4])
    assert output == [
        dict(utt_id="3", ref="A", hyp="C"),
        dict(utt_id="4", ref="B", hyp=""),
    ]
    with pytest.raises(ValueError):
        build_output([{"text": "A"}], [], [0])


def test_model_matches_native_espnet2_initial_state():
    """Compare every parameter in the full Conformer under the same seed."""
    from espnet3.utils.task_utils import get_espnet_model

    original = yaml.safe_load(
        (ROOT / "egs2/voices/asr1/conf/train_asr_conformer.yaml").read_text()
    )
    config = load_and_merge_config(
        RECIPE / "conf/training_conformer.yaml", "training.yaml"
    )
    migrated = OmegaConf.to_container(config.model, resolve=True)
    tokens = ["<blank>", "<unk>", *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "▁", "<sos/eos>"]
    for arguments in (original, migrated):
        # ASRTask's parser converts the YAML string "none" to Python None.
        arguments.update(
            token_list=tokens, normalize=None, normalize_conf={}, init=None
        )
    torch.manual_seed(0)
    before = get_espnet_model("espnet2.tasks.asr.ASRTask", original).state_dict()
    torch.manual_seed(0)
    after = get_espnet_model("espnet3.systems.asr.task.ASRTask", migrated).state_dict()
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_complete_stage_pipeline_on_fixture(corpus, monkeypatch):
    """Exercise all six shared stages with a tiny CPU model and real WAV files."""
    from egs3.TEMPLATE.asr.run import DEFAULT_STAGES, build_parser, main
    from egs3.voices.asr.src import tokenizer as tokenizer_module
    from egs3.voices.asr.src.system import VoicesSystem

    recipe, source = corpus
    monkeypatch.setattr(
        tokenizer_module,
        "gather_training_text",
        lambda **_: [" ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ")] * 5,
    )
    training = load_and_merge_config(
        RECIPE / "conf/training_devkit.yaml", "training.yaml", resolve=False
    )
    training.recipe_dir = str(RECIPE)
    training.data_dir = str(recipe / "data")
    training.exp_dir = str(recipe / "experiment")
    training.stats_dir = str(recipe / "statistics")
    training.create_dataset = {"recipe_dir": str(recipe), "source_dir": str(source)}
    for split in ("train", "valid"):
        training.dataset[split][0].data_src = "egs3.voices.asr.dataset"
        training.dataset[split][0].data_src_args.recipe_dir = str(recipe)
        training.dataloader[split].iter_factory.num_workers = 0
    training.tokenizer.vocab_size = 30
    training.tokenizer.save_path = str(recipe / "tokenizer")
    training.model.encoder_conf.update(
        output_size=16, attention_heads=2, linear_units=32, num_blocks=1
    )
    training.model.decoder_conf.update(attention_heads=2, linear_units=32, num_blocks=1)
    training.num_device = 1
    training.espnet2_compat.accum_grad = 1
    training.trainer.update(
        accelerator="cpu",
        devices=1,
        # This single-process fixture must not leave a DDP group in pytest.
        strategy="auto",
        precision="32-true",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
    )
    training.dataloader.train.iter_factory.batches.batch_bins = 100000
    inference = load_and_merge_config(
        RECIPE / "conf/inference.yaml", "inference.yaml", resolve=False
    )
    inference.recipe_dir = str(RECIPE)
    inference.exp_dir = str(recipe / "experiment")
    inference.inference_dir = str(recipe / "inference")
    inference.batch_size = 1
    inference.model.update(
        beam_size=2, maxlenratio=0.1, lm_weight=0.0, lm_file=None, lm_train_config=None
    )
    for entry in inference.dataset.test:
        entry.data_src = "egs3.voices.asr.dataset"
        entry.data_src_args.recipe_dir = str(recipe)
        entry.data_src_args.limit = 1
    metrics = load_and_merge_config(
        RECIPE / "conf/metrics.yaml", "metrics.yaml", resolve=False
    )
    metrics.metrics[2].metric.bpemodel = str(recipe / "tokenizer/unigram.model")
    metrics_path = recipe / "metrics.yaml"
    OmegaConf.save(metrics, metrics_path)
    training_path = recipe / "training.yaml"
    inference_path = recipe / "inference.yaml"
    OmegaConf.save(training, training_path)
    OmegaConf.save(inference, inference_path)
    args = build_parser(DEFAULT_STAGES).parse_args(
        [
            "--stages",
            "create_dataset",
            "train_tokenizer",
            "collect_stats",
            "train",
            "infer",
            "measure",
            "--training_config",
            str(training_path),
            "--inference_config",
            str(inference_path),
            "--metrics_config",
            str(metrics_path),
        ]
    )
    main(args, VoicesSystem, DEFAULT_STAGES)
    assert (recipe / "experiment/valid.acc.final_ave.pth").is_file()
    assert (recipe / "inference/test/hyp.scp").is_file()
    assert (recipe / "inference/metrics.json").is_file()


def test_missing_clean_source_is_not_silently_accepted(corpus):
    """A complete distant subset also requires every corresponding clean source."""
    recipe, source = corpus
    next((source / "source-16k/train").rglob("*.wav")).unlink()
    with pytest.raises(ValueError, match="Source/distant utterance sets differ"):
        DatasetBuilder().build(recipe_dir=recipe)
    assert not DatasetBuilder().is_built(recipe_dir=recipe)


def test_build_marker_with_real_builder_configuration(tmp_path, monkeypatch):
    """Persist the production YAML settings without replacing their container type."""
    rows = [
        dict(
            utt_id=f"{speaker:04d}_recording",
            path="/unused.wav",
            text="WORDS",
            speaker=f"{speaker:04d}",
            source_id=f"source_{speaker}",
            condition="distant",
            samples=16000,
        )
        for speaker in range(1, 13)
    ]
    monkeypatch.setattr(module, "_load_transcripts", lambda _: {})
    monkeypatch.setattr(
        module,
        "_scan_recordings",
        lambda _, split, __, corpus: rows[:-1] if split == "train" else rows[-1:],
    )
    builder = DatasetBuilder()
    builder.build(recipe_dir=tmp_path)
    metadata = json.loads((tmp_path / "data/manifest/build.json").read_text())
    assert metadata["builder"]["expected_distant_counts"] == {
        "train": 12800,
        "test": 6400,
    }
    assert metadata["counts"] == {"train": 1, "valid": 10, "test": 1}
    assert builder.is_built(recipe_dir=tmp_path)


def test_tokenizer_and_lm_keep_pre_filter_text(corpus):
    """Duration filtering must not remove source tokenizer or LM training text."""
    recipe, source = corpus
    path = sorted((source / "source-16k/train").rglob("*.wav"))[-1]
    sf.write(path, np.zeros(480000), 16000)
    DatasetBuilder().build(recipe_dir=recipe)
    assert len(Dataset("train", recipe_dir=recipe)) == 5
    assert len(gather_training_text(recipe / "data/manifest/tokenizer_train.txt")) == 6
    assert len((recipe / "data/lm/train.txt").read_text().splitlines()) == 6


def test_full_corpus_mode_with_existing_layout(corpus, monkeypatch):
    """Apply the full-corpus mode to a local layout without a large download."""
    recipe, source = corpus
    monkeypatch.setitem(
        module._BUILDER_CFG, "full_distant_counts", {"train": 8, "test": 2}
    )
    builder = DatasetBuilder()
    builder.prepare_source(recipe_dir=recipe, source_dir=source, corpus="full")
    builder.build(recipe_dir=recipe, source_dir=source, corpus="full")
    assert builder.is_built(recipe_dir=recipe, source_dir=source, corpus="full")
    assert not builder.is_built(recipe_dir=recipe, source_dir=source, corpus="devkit")
    assert len(Dataset("train", recipe_dir=recipe)) == 6
