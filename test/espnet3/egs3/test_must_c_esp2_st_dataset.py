"""Language selection in the MuST-C ST recipe.

The recipe trains on one language pair, and three places have to agree on
which: the builder's default, the ``create_dataset`` block the stage forwards,
and the cache the Dataset reads back. Each test here stands for a way they
came apart.
"""

from pathlib import Path

import pytest
import yaml
from datasets import Dataset as HFDataset

from egs3.must_c.esp2_st.dataset import builder
from egs3.must_c.esp2_st.dataset.dataset import _assert_cache_language

RECIPE_DIR = Path(builder.__file__).resolve().parents[1]


def _install_pair(root: Path, tgt_lang: str) -> None:
    """Write the files ``missing_required_splits`` looks for, and nothing else."""
    for split in builder.REQUIRED_SPLITS:
        split_dir = root / f"{builder.SRC_LANG}-{tgt_lang}" / "data" / split
        (split_dir / "wav").mkdir(parents=True, exist_ok=True)
        text_dir = split_dir / "txt"
        text_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            f"{split}.yaml",
            f"{split}.{builder.SRC_LANG}",
            f"{split}.{tgt_lang}",
        ):
            (text_dir / name).write_text("", encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    """A corpus root holding two complete pairs."""
    root = tmp_path / "must-c"
    _install_pair(root, "de")
    _install_pair(root, "fr")
    return root


def _cache(tgt_lang=None, rows=1, column=True):
    """A stand-in for one split of the HF cache."""
    if rows == 0:
        return HFDataset.from_dict({"tgt_lang": []} if column else {"utt_id": []})
    data = {"utt_id": [f"u{i}" for i in range(rows)]}
    if column:
        data["tgt_lang"] = [tgt_lang] * rows
    return HFDataset.from_dict(data)


def test_default_pair_is_the_one_the_recipe_trains_on():
    # `all` used to be the default and could not be built at all.
    assert builder.TGT_LANG == "de"
    assert builder._resolve_tgt_lang(None) == "de"
    assert builder._resolve_tgt_lang("fr") == "fr"


def test_create_dataset_stage_is_given_the_pair():
    # BaseSystem.create_dataset forwards this block and nothing else, so the
    # `tgt_lang` in the dataset entries never reaches the builder.
    config = yaml.safe_load(
        (RECIPE_DIR / "conf" / "tuning" / "train_st_conformer.yaml").read_text()
    )
    assert config["create_dataset"]["tgt_lang"] == "de"


def test_missing_pair_names_the_installed_ones(tmp_path, corpus):
    with pytest.raises(FileNotFoundError) as excinfo:
        builder.resolve_source_root(tmp_path, corpus, "xx")
    message = str(excinfo.value)
    assert "en-xx" in message
    assert "Installed pairs: de, fr." in message


def test_aggregate_language_is_not_a_special_case(tmp_path, corpus):
    # `all` is just a pair nobody installed; it must not resolve to a literal
    # en-all directory, nor be answered with a different kind of error.
    with pytest.raises(FileNotFoundError) as excinfo:
        builder.resolve_source_root(tmp_path, corpus, "all")
    assert "en-all" in str(excinfo.value)
    assert "Installed pairs: de, fr." in str(excinfo.value)


def test_source_checks_follow_the_selected_pair(tmp_path, corpus):
    instance = builder.MustCSTBuilder()
    assert instance.is_source_prepared(tmp_path, corpus, tgt_lang="de") is True
    assert instance.is_source_prepared(tmp_path, corpus, tgt_lang="xx") is False
    assert instance.is_source_prepared(tmp_path, corpus, tgt_lang="all") is False
    instance.prepare_source(tmp_path, corpus, tgt_lang="de")


def test_cache_of_another_pair_is_refused():
    with pytest.raises(ValueError, match="holds en-de"):
        _assert_cache_language(
            _cache("de"), "fr", {"enabled": True, "cache_dir": "data/hf/en_de"}
        )


def test_cache_from_the_environment_is_refused_with_a_message(monkeypatch):
    # `cache` is None on this path: _load_hf_cache builds its own dict from
    # the variable, so reading cache["cache_dir"] raised AttributeError and
    # hid the mismatch it was added to report.
    monkeypatch.setenv("EGS3_HF_CACHE_DIR", "/tmp/en_de")
    with pytest.raises(ValueError, match="holds en-de"):
        _assert_cache_language(_cache("de"), "fr", None)


@pytest.mark.parametrize(
    "cached, tgt_lang",
    [
        (_cache("de"), "de"),
        (_cache("de"), None),
        (_cache(rows=1, column=False), "fr"),
        (_cache(rows=0), "fr"),
    ],
    ids=[
        "matching pair",
        "no pair requested",
        "cache predates the column",
        "empty cache",
    ],
)
def test_cache_is_accepted(cached, tgt_lang):
    _assert_cache_language(cached, tgt_lang, {"cache_dir": "data/hf/en_de"})


def _build_cache(root, tgt_lang, with_language=True):
    """Write the splits and columns ``is_built`` looks for."""
    index = root / "hf_audio_index"
    for split in builder._HF_CACHE_SPLITS:
        data = {
            "audio_path": ["a.wav"],
            "src_text": ["hello"],
            "tgt_text": ["hallo"],
            "offset": [0.0],
            "duration": [1.0],
        }
        if with_language:
            data["tgt_lang"] = [tgt_lang]
        HFDataset.from_dict(data).save_to_disk(str(index / split))
    return {"enabled": True, "backend": "hf", "cache_dir": str(root)}


def test_is_built_refuses_a_cache_built_for_another_pair(tmp_path):
    # create_dataset skips build() when is_built() is true, so a cache of the
    # wrong pair was reported as ready and the mismatch only surfaced later,
    # when the Dataset read it.
    cache = _build_cache(tmp_path / "en_fr", "fr")
    with pytest.raises(ValueError, match="holds en-fr"):
        builder.MustCSTBuilder().is_built(tmp_path, cache, tgt_lang="de")


def test_is_built_accepts_the_pair_it_was_built_for(tmp_path):
    cache = _build_cache(tmp_path / "en_fr", "fr")
    assert builder.MustCSTBuilder().is_built(tmp_path, cache, tgt_lang="fr") is True


def test_is_built_accepts_a_cache_predating_the_language_column(tmp_path):
    cache = _build_cache(tmp_path / "en_de", "de", with_language=False)
    assert builder.MustCSTBuilder().is_built(tmp_path, cache, tgt_lang="de") is True


def test_is_built_is_false_when_there_is_no_cache(tmp_path):
    cache = {"enabled": True, "backend": "hf", "cache_dir": str(tmp_path / "absent")}
    assert builder.MustCSTBuilder().is_built(tmp_path, cache, tgt_lang="de") is False
