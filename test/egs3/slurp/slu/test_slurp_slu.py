"""Tests for the ESPnet3 SLURP SLU recipe."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.slurp.slu.dataset.builder import (
    SlurpBuilder,
    get_manifest_path,
    normalize_transcript,
    read_manifest,
)
from egs3.slurp.slu.dataset.dataset import SlurpDataset
from egs3.slurp.slu.src.inference import build_output, split_intent
from egs3.slurp.slu.src.metrics import IntentAccuracy
from egs3.slurp.slu.src.tokenizer import gather_training_text, read_intent_labels

_PROMPTS = {
    "train": [("audio", "volume_mute", "Turn off the speakers.", "1")],
    "train_synthetic": [("audio", "volume_mute", "MUTE the Speakers", "2")],
    "devel": [("calendar", "query", "what's on my agenda @ noon", "3")],
    "test": [("news", "query", "read me the news, please", "4")],
}


def _write_prompt_file(path: Path, prompts, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as prompt_file:
        for scenario, action, sentence, recording in prompts:
            prompt_file.write(
                json.dumps(
                    {
                        "scenario": scenario,
                        "action": action,
                        "sentence": sentence,
                        "recordings": [{"file": f"audio-{prefix}{recording}.flac"}],
                    }
                )
                + "\n"
            )


@pytest.fixture()
def slurp_root(tmp_path: Path) -> Path:
    """Create a miniature SLURP corpus with real and synthetic audio."""
    root = tmp_path / "slurp"
    metadata = {}
    for split, prompts in _PROMPTS.items():
        prefix = "s" if split == "train_synthetic" else "r"
        _write_prompt_file(
            root / "dataset" / "slurp" / f"{split}.jsonl", prompts, prefix
        )
        audio_dir = (
            root
            / "audio"
            / ("slurp_synth" if split == "train_synthetic" else "slurp_real")
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        for _, _, _, recording in prompts:
            name = f"audio-{prefix}{recording}.flac"
            sf.write(audio_dir / name, np.zeros(1600, dtype=np.float32), 16000)
            metadata[f"{split}-{recording}"] = {
                "recordings": {name: {"usrid": f"user{recording}"}}
            }
    with (root / "dataset" / "slurp" / "metadata.json").open(
        "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(metadata, metadata_file)
    return root


@pytest.fixture()
def recipe_dir(tmp_path: Path, slurp_root: Path) -> Path:
    """Build the recipe manifests from the miniature corpus."""
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    SlurpBuilder().build(recipe_dir=recipe, source_dir=slurp_root)
    return recipe


def test_normalize_transcript_matches_egs2_rules():
    assert normalize_transcript("mail  @ home, now.") == "mail at home now"
    assert normalize_transcript("#tag <unk>") == "hashtag tag unknown"
    assert normalize_transcript("Mute IT", lowercase=True) == "mute it"


def test_builder_writes_manifests_and_intent_list(recipe_dir: Path):
    rows = read_manifest(get_manifest_path(recipe_dir, "train"))
    assert len(rows) == 1
    assert rows[0]["intent"] == "audio_volume_mute"
    assert rows[0]["transcript"] == "Turn off the speakers"
    assert rows[0]["utt_id"] == "slurp_user1_r1"

    synthetic = read_manifest(get_manifest_path(recipe_dir, "train_synthetic"))
    assert synthetic[0]["transcript"] == "mute the speakers"
    assert synthetic[0]["utt_id"] == "slurp_synthetic_s2"

    # Only the training splits contribute intent labels.
    assert read_intent_labels(recipe_dir) == ["audio_volume_mute"]


def test_builder_leaves_no_temporary_files(recipe_dir: Path):
    """Manifests are written through a temp file that must be renamed away."""
    leftovers = sorted(p.name for p in (recipe_dir / "data" / "manifest").glob(".*"))
    assert leftovers == []


def test_builder_is_built_reports_readiness(tmp_path: Path, slurp_root: Path):
    builder = SlurpBuilder()
    recipe = tmp_path / "empty_recipe"
    recipe.mkdir()
    assert builder.is_source_prepared(recipe_dir=recipe, source_dir=slurp_root)
    assert not builder.is_built(recipe_dir=recipe, source_dir=slurp_root)
    builder.build(recipe_dir=recipe, source_dir=slurp_root)
    assert builder.is_built(recipe_dir=recipe, source_dir=slurp_root)


def test_builder_rejects_missing_corpus(tmp_path: Path):
    recipe = tmp_path / "recipe_without_corpus"
    recipe.mkdir()
    builder = SlurpBuilder()
    assert not builder.is_source_prepared(recipe_dir=recipe)
    with pytest.raises(FileNotFoundError, match="SLURP source not found"):
        builder.prepare_source(recipe_dir=recipe)


def test_dataset_returns_only_supported_fields(recipe_dir: Path, slurp_root: Path):
    """A recipe sample must never carry utt_id: it breaks downstream collation."""
    dataset = SlurpDataset(split="devel", recipe_dir=recipe_dir, source_dir=slurp_root)
    sample = dataset[0]
    assert sorted(sample) == ["speech", "text"]
    assert sample["text"] == "calendar_query what's on my agenda at noon"
    assert sample["speech"].dtype == np.float32


def test_dataset_rejects_unknown_split(recipe_dir: Path, slurp_root: Path):
    with pytest.raises(ValueError, match="Unknown split"):
        SlurpDataset(split="eval", recipe_dir=recipe_dir, source_dir=slurp_root)


def test_gather_training_text_returns_transcripts_only(
    recipe_dir: Path, slurp_root: Path
):
    texts = gather_training_text(recipe_dir=recipe_dir, source_dir=slurp_root)
    assert texts == ["Turn off the speakers", "mute the speakers"]


def test_read_intent_labels_requires_create_dataset(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Intent label list not found"):
        read_intent_labels(tmp_path)


def test_split_intent_handles_partial_text():
    assert split_intent("audio_volume_mute mute it") == (
        "audio_volume_mute",
        "mute it",
    )
    assert split_intent("audio_volume_mute") == ("audio_volume_mute", "")
    assert split_intent("  ") == ("", "")


def test_build_output_splits_intent_and_transcript():
    data = {"text": "news_query read the news"}
    output = build_output(data, [["news_query read news"]], 0)
    assert output == {
        "utt_id": "0",
        "hyp": "news_query read news",
        "ref": "news_query read the news",
        "hyp_intent": "news_query",
        "ref_intent": "news_query",
        "hyp_transcript": "read news",
        "ref_transcript": "read the news",
    }


def test_build_output_handles_batches():
    """`inference.yaml` inherits batch_size 4, so batched calls must work."""
    data = [{"text": "news_query a"}, {"text": "qa_maths b"}]
    model_output = [[["news_query a"]], [["qa_maths c"]]]
    outputs = build_output(data, model_output, [0, 1])
    assert [o["utt_id"] for o in outputs] == ["0", "1"]
    assert [o["hyp_intent"] for o in outputs] == ["news_query", "qa_maths"]
    assert outputs[1]["hyp_transcript"] == "c"


def test_intent_accuracy_scores_and_reports_errors(tmp_path: Path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "ref_intent.scp").write_text(
        "0 news_query\n1 audio_volume_mute\n", encoding="utf-8"
    )
    (test_dir / "hyp_intent.scp").write_text(
        "0 news_query\n1 calendar_query\n", encoding="utf-8"
    )

    metric = IntentAccuracy()
    result = metric(
        {
            "ref_intent": test_dir / "ref_intent.scp",
            "hyp_intent": test_dir / "hyp_intent.scp",
        },
        "test",
        tmp_path,
    )

    assert result == {"IntentAccuracy": 50.0}
    errors = (test_dir / "intent_errors").read_text(encoding="utf-8").splitlines()
    assert errors[1] == "1\taudio_volume_mute\tcalendar_query"
