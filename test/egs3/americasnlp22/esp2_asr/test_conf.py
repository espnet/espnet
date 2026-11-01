"""Tests for the AmericasNLP 2022 configs and their ASR TEMPLATE inheritance."""

from pathlib import Path

import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

from egs3.americasnlp22.esp2_asr.dataset.dataset import AmericasNLP22Dataset
from espnet3.utils.config_utils import load_and_merge_config

_CONF_DIR = Path(__file__).resolve().parents[4] / "egs3/americasnlp22/esp2_asr/conf"
_LANGS = ("bzd", "gn", "gvc", "quy", "tav")


def _load(role: str, lang: str | None = None):
    """Load a recipe config merged onto the ASR TEMPLATE, as ``launch()`` does.

    With ``lang``, the per-language ``conf/tuning/<role>_<lang>.yaml``.
    """
    path = _CONF_DIR / (f"tuning/{role}_{lang}.yaml" if lang else f"{role}.yaml")
    return load_and_merge_config(
        path, config_name=f"{role}.yaml", default_package="egs3.TEMPLATE.esp2_asr"
    )


@pytest.mark.parametrize("lang", _LANGS)
def test_training_config_selects_language(lang) -> None:
    config = _load("training", lang)
    assert config.exp_tag == f"training_{lang}"
    assert config.create_dataset.lang == lang
    for split in ("train", "valid"):
        [entry] = config.dataset[split]
        assert entry.data_src_args.lang == lang
        assert entry.data_src_args.min_wav_duration == 0.1
        assert entry.data_src_args.max_wav_duration == 38
    assert config.tokenizer.text_builder.lang == lang
    assert config.tokenizer.model_type == "unigram"
    assert config.tokenizer.vocab_size == 100
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
    args = test_set.data_src_args
    assert (args.lang, args.split) == (lang, "dev")
    assert (args.min_wav_duration, args.max_wav_duration) == (0.1, 38)
    # The recipe sets egs2's decoding; model target, runner, contract,
    # input_key and batch_size come from the TEMPLATE.
    model = config.model
    assert (model.beam_size, model.ctc_weight, model.lm_weight) == (5, 1.0, 0.0)
    assert model.asr_model_file.endswith("/valid.cer_ctc.ave_1best.pth")
    assert model._target_ == "espnet2.bin.asr_inference.Speech2Text"
    assert config.runner._target_.endswith(".esp2_asr.inference.ASRInferenceRunner")
    assert config.input_key == "speech"
    assert list(config.contract.inputs) == ["speech"]
    assert list(config.contract.outputs) == ["text"]
    assert config.batch_size == 4


def test_metrics_config_scores_text_against_dataset_text() -> None:
    config = _load("metrics")
    targets = [entry.metric._target_ for entry in config.metrics]
    assert targets == [
        "espnet3.systems.esp2_asr.metrics.wer.WER",
        "espnet3.systems.esp2_asr.metrics.cer.CER",
    ]
    for entry in config.metrics:
        assert entry.metric.clean_types is None
        assert dict(entry.inputs) == {"ref": "dataset:text", "hyp": "text"}


def test_dev_item_passes_the_inherited_inference_contract(
    tmp_path, corpus_factory
) -> None:
    corpus_factory(tmp_path)
    dataset = AmericasNLP22Dataset(
        lang="bzd", split="dev", recipe_dir=tmp_path, source_dir=tmp_path
    )
    config = _load("inference", "bzd")
    calls = []

    def speech2text(speech):
        """Return one Speech2Text-shaped n-best list per waveform."""
        calls.append(speech)
        return [[(f"{len(wave)} samples", [], [], None)] for wave in speech]

    outputs = get_class(config.runner._target_).forward(
        [0],
        dataset=dataset,
        model=speech2text,
        input_key=config.input_key,
        inference_contract=OmegaConf.to_container(config.contract, resolve=True),
    )
    # The item has no utt_id, so the row id is its index.
    assert outputs == [{"text": "8000 samples", "utt_id": "0"}]
    assert calls[0][0].shape == (8000,)
