"""Tests for the OWSM system's tokenizer stage.

Only the behaviour that differs from ``ASRSystem`` is covered here: reserving
the OWSM special symbols, guarding the vocabulary size against them, and
reusing an already-gathered training text instead of failing on it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.systems.owsm import OWSMSystem

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEM_SOURCE = REPO_ROOT / "espnet3" / "systems" / "owsm" / "system.py"


def _config(tmp_path: Path, **tokenizer):
    base = dict(
        vocab_size=100,
        model_type="bpe",
        save_path=str(tmp_path / "bpe"),
        nlsyms=["<na>", "<nospeech>", "<eng>", "<asr>"],
    )
    base.update(tokenizer)
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "data_dir": str(tmp_path / "data"),
            "tokenizer": base,
        }
    )


def test_package_does_not_depend_on_other_system_packages():
    """It must stand alone; only base and its own tokenizers are allowed."""
    tree = ast.parse(SYSTEM_SOURCE.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
    foreign = [
        module
        for module in modules
        if module.startswith("espnet3.systems.")
        and not module.startswith(("espnet3.systems.base", "espnet3.systems.owsm"))
    ]
    assert not foreign, foreign


def test_missing_nlsyms_is_rejected(tmp_path):
    config = _config(tmp_path)
    del config.tokenizer.nlsyms
    system = OWSMSystem(training_config=config)
    with pytest.raises(RuntimeError, match="nlsyms"):
        system.train_tokenizer()


def test_nlsyms_can_come_from_a_file(tmp_path):
    symbols = tmp_path / "nlsyms.txt"
    symbols.write_text("<na>\n<nospeech>\n<eng>\n<asr>\n", encoding="utf-8")
    system = OWSMSystem(training_config=_config(tmp_path, nlsyms=str(symbols)))
    assert system._special_symbols() == ["<na>", "<nospeech>", "<eng>", "<asr>"]


def test_vocab_size_must_exceed_the_reserved_symbols(tmp_path):
    """The symbols occupy the vocabulary before a single piece is learned."""
    system = OWSMSystem(training_config=_config(tmp_path, vocab_size=4))
    with pytest.raises(RuntimeError, match="must exceed"):
        system.train_tokenizer()


def test_existing_training_text_is_reused(tmp_path):
    """The gather walks every corpus, so an interrupted run must not redo it."""
    config = _config(tmp_path)
    train_file = tmp_path / "data" / "train_tokenizer" / "train.txt"
    train_file.parent.mkdir(parents=True)
    train_file.write_text("<eng><asr><0.00> hello<1.00>", encoding="utf-8")

    system = OWSMSystem(training_config=config)
    assert system._gather_text(train_file) == ["<eng><asr><0.00> hello<1.00>"]


def test_reuse_can_be_turned_off(tmp_path):
    config = _config(tmp_path, reuse_existing_text=False)
    train_file = tmp_path / "data" / "train_tokenizer" / "train.txt"
    train_file.parent.mkdir(parents=True)
    train_file.write_text("anything", encoding="utf-8")

    system = OWSMSystem(training_config=config)
    with pytest.raises(RuntimeError, match="already exists"):
        system._gather_text(train_file)


def test_text_builder_is_instantiated_by_hydra(tmp_path, monkeypatch):
    """The builder is named by _target_ and called with the remaining keys."""
    module = tmp_path / "owsm_fake_builder.py"
    module.write_text(
        "def build(corpora):\n"
        "    return [f'<eng><asr><0.00> {c}<1.00>' for c in corpora]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _config(
        tmp_path,
        text_builder={
            "_target_": "owsm_fake_builder.build",
            # A list must reach the builder as a list, not a ListConfig.
            "corpora": ["spgispeech", "must_c"],
        },
    )
    system = OWSMSystem(training_config=config)
    train_file = tmp_path / "data" / "train_tokenizer" / "train.txt"

    texts = system._gather_text(train_file)

    assert texts == [
        "<eng><asr><0.00> spgispeech<1.00>",
        "<eng><asr><0.00> must_c<1.00>",
    ]
    # The gather walks every corpus, so its result is cached for a rerun.
    assert train_file.read_text(encoding="utf-8") == "\n".join(texts)


def test_missing_text_builder_target_is_rejected(tmp_path):
    system = OWSMSystem(training_config=_config(tmp_path, text_builder={}))
    with pytest.raises(RuntimeError, match="_target_"):
        system._gather_text(tmp_path / "absent.txt")


def test_symbols_survive_tokenization(tmp_path):
    """End to end: the tags must come back as single pieces, not characters."""
    spm = pytest.importorskip("sentencepiece")

    symbols = ["<na>", "<nospeech>", "<eng>", "<asr>", "<st_deu>", "<0.00>", "<1.00>"]
    config = _config(tmp_path, vocab_size=120, nlsyms=symbols)
    train_file = tmp_path / "data" / "train_tokenizer" / "train.txt"
    train_file.parent.mkdir(parents=True)
    # Enough distinct text for SentencePiece to reach the requested vocabulary.
    train_file.write_text(
        "\n".join(
            f"<eng><asr><0.00> the quick brown fox number {i} jumps over lazy "
            f"dogs again and again<1.00>"
            for i in range(200)
        ),
        encoding="utf-8",
    )

    OWSMSystem(training_config=config).train_tokenizer()

    model = Path(config.tokenizer.save_path) / "bpe.model"
    assert model.is_file()
    processor = spm.SentencePieceProcessor(model_file=str(model))
    pieces = processor.encode("<eng><asr><0.00> the fox<1.00>", out_type=str)
    for symbol in ("<eng>", "<asr>", "<0.00>", "<1.00>"):
        assert symbol in pieces, (symbol, pieces)


def _stats_config(tmp_path, **tokenizer):
    config = _config(tmp_path, **tokenizer)
    config.stats_dir = str(tmp_path / "stats")
    return config


def _write_tokens(tmp_path, count):
    save = Path(tmp_path / "bpe")
    save.mkdir(parents=True, exist_ok=True)
    (save / "tokens.txt").write_text(
        "\n".join(f"tok{i}" for i in range(count)), encoding="utf-8"
    )


def _write_shapes(tmp_path, mode, streams, body="a 7\n"):
    """collect_stats always writes both modes, so the fixtures must too."""
    stats = Path(tmp_path / "stats" / mode)
    stats.mkdir(parents=True, exist_ok=True)
    for stream in streams:
        (stats / f"{stream}_shape").write_text(body, encoding="utf-8")
    return stats


def test_text_shapes_become_two_dimensional(tmp_path):
    """A bare L contributes L*1 to a numel budget, i.e. nothing."""
    _write_tokens(tmp_path, 50)
    system = OWSMSystem(training_config=_stats_config(tmp_path))
    streams = ("text", "text_prev", "text_ctc")
    stats = _write_shapes(tmp_path, "train", streams, "a 7\nb 9\n")
    _write_shapes(tmp_path, "valid", streams, "a 7\nb 9\n")
    (stats / "feats_shape").write_text("a 100,128\nb 200,128\n", encoding="utf-8")

    system._append_vocab_size_to_text_shapes()

    # text and text_prev share the decoder logits tensor, so both scale with V.
    for stream in ("text", "text_prev"):
        assert (stats / f"{stream}_shape").read_text().split() == [
            "a", "7,50", "b", "9,50",
        ]
    # text_ctc does not: CTC projects the encoder, so its length never meets V.
    assert (stats / "text_ctc_shape").read_text() == "a 7\nb 9\n"
    # The speech shape must be left alone; it is already 2-D and is the key the
    # sampler sorts on.
    assert (stats / "feats_shape").read_text() == "a 100,128\nb 200,128\n"


def test_appending_vocab_size_is_idempotent(tmp_path):
    _write_tokens(tmp_path, 50)
    system = OWSMSystem(training_config=_stats_config(tmp_path))
    _write_shapes(tmp_path, "train", ("text", "text_prev"))
    stats = _write_shapes(tmp_path, "valid", ("text", "text_prev"))

    system._append_vocab_size_to_text_shapes()
    system._append_vocab_size_to_text_shapes()

    assert (stats / "text_shape").read_text() == "a 7,50\n"


def test_text_ctc_can_be_scaled_by_request(tmp_path):
    """The default leaves text_ctc 1-D, but a config may opt it in."""
    _write_tokens(tmp_path, 50)
    config = _stats_config(tmp_path)
    config.tokenizer.vocab_scaled_shapes = ["text", "text_prev", "text_ctc"]
    system = OWSMSystem(training_config=config)
    streams = ("text", "text_prev", "text_ctc")
    stats = _write_shapes(tmp_path, "train", streams)
    _write_shapes(tmp_path, "valid", streams)

    system._append_vocab_size_to_text_shapes()

    assert (stats / "text_ctc_shape").read_text() == "a 7,50\n"


def test_a_configured_stream_without_a_shape_file_is_rejected(tmp_path):
    """Skipping it would hide that stream's cost from the batcher until OOM."""
    _write_tokens(tmp_path, 50)
    config = _stats_config(tmp_path)
    config.tokenizer.vocab_scaled_shapes = ["text", "text_prev"]
    system = OWSMSystem(training_config=config)
    _write_shapes(tmp_path, "train", ("text",))
    _write_shapes(tmp_path, "valid", ("text", "text_prev"))

    with pytest.raises(RuntimeError, match="text_prev"):
        system._append_vocab_size_to_text_shapes()


def test_sampling_is_seeded_and_spans_the_whole_stream(tmp_path):
    """A prefix would bias the vocabulary toward whichever corpus is written first."""
    from espnet3.systems.owsm.system import _reservoir_sample

    lines = [f"line {i}" for i in range(1000)]
    first, seen = _reservoir_sample(iter(lines), 50, seed=0)
    again, _ = _reservoir_sample(iter(lines), 50, seed=0)
    other, _ = _reservoir_sample(iter(lines), 50, seed=1)

    assert seen == 1000 and len(first) == 50
    assert first == again, "the same seed must give the same vocabulary"
    assert first != other
    # Drawn from the whole stream, not the head.
    indices = [int(line.split()[1]) for line in first]
    assert max(indices) > 500, indices


def test_no_sample_size_keeps_everything(tmp_path):
    from espnet3.systems.owsm.system import _reservoir_sample

    kept, seen = _reservoir_sample(iter(["a", "b", "c"]), None, seed=0)
    assert kept == ["a", "b", "c"] and seen == 3


def test_gather_samples_and_reports(tmp_path, monkeypatch, caplog):
    module = tmp_path / "owsm_big_builder.py"
    module.write_text(
        "def build():\n    return (f'<eng><asr> line {i}' for i in range(500))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _config(
        tmp_path,
        sample_size=40,
        sample_seed=7,
        text_builder={"_target_": "owsm_big_builder.build"},
    )
    system = OWSMSystem(training_config=config)
    train_file = tmp_path / "data" / "train_tokenizer" / "train.txt"

    with caplog.at_level("INFO"):
        texts = system._gather_text(train_file)

    assert len(texts) == 40
    assert "Sampled 40 of 500 lines (seed 7)" in caplog.text
    # Only the sample is written, not the stream it came from.
    assert len(train_file.read_text(encoding="utf-8").splitlines()) == 40


def test_nlsyms_can_be_a_callable(tmp_path):
    """The OWSM inventory is over 1700 symbols; a config cannot inline it."""
    module = tmp_path / "owsm_syms.py"
    module.write_text(
        "def symbols():\n    return ['<na>', '<nospeech>', '<eng>', '<asr>']\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        config = _config(tmp_path, nlsyms={"_target_": "owsm_syms.symbols"})
        system = OWSMSystem(training_config=config)
        assert system._special_symbols() == ["<na>", "<nospeech>", "<eng>", "<asr>"]
    finally:
        sys.path.remove(str(tmp_path))
