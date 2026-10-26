"""Tests for the AmericasNLP 2022 per-language config composition."""

from pathlib import Path

import pytest

from espnet3.utils.config_utils import load_and_merge_config

_CONF_DIR = Path(__file__).resolve().parents[4] / "egs3/americasnlp22/esp2_asr/conf"
_LANGS = ("bzd", "gn", "gvc", "quy", "tav")


def _load(role: str, lang: str):
    """Load ``conf/tuning/<role>_<lang>.yaml`` the way ``launch()`` does."""
    return load_and_merge_config(
        _CONF_DIR / "tuning" / f"{role}_{lang}.yaml",
        config_name=f"{role}.yaml",
        default_package="egs3.TEMPLATE.esp2_asr",
    )


@pytest.mark.parametrize("lang", _LANGS)
def test_training_config_selects_language(lang) -> None:
    config = _load("training", lang)
    assert config.exp_tag == f"training_{lang}"
    assert config.create_dataset.lang == lang
    for split in ("train", "valid"):
        [entry] = config.dataset[split]
        assert entry.data_src_args.lang == lang
    assert config.tokenizer.text_builder.lang == lang
    assert config.dataset.preprocessor.bpemodel == (
        f"{config.tokenizer.save_path}/unigram.model"
    )


def test_languages_do_not_share_outputs() -> None:
    # Shape files and tokenizer text are per language; sharing them would
    # make the second language read the first one's statistics or fail on
    # the already written tokenizer text.
    configs = [_load("training", lang) for lang in _LANGS]
    for get in (
        lambda c: c.exp_dir,
        lambda c: c.stats_dir,
        lambda c: c.tokenizer.save_path,
        lambda c: c.tokenizer.train_file,
    ):
        assert len({get(c) for c in configs}) == len(_LANGS)


@pytest.mark.parametrize("lang", _LANGS)
def test_inference_config_scores_dev_set(lang) -> None:
    config = _load("inference", lang)
    [test_set] = config.dataset.test
    assert test_set.name == f"dev_{lang}"
    assert test_set.data_src_args.lang == lang
    assert test_set.data_src_args.split == "dev"
