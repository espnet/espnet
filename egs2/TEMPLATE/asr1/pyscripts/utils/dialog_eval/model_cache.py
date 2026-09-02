"""Process-wide cache for the evaluation models used by ``dialog_eval``.

Every helper in this package builds its scoring models inside the function that
scores a single utterance, so the models are reconstructed on every call. The
arguments passed to each ``*_setup`` are compile-time constants, so the results
can be built once and reused, which is what this module does.

Two helpers, :mod:`ASR_WER` and :mod:`TTS_intelligibility`, additionally request
the *same* three ASR systems. Routing both through here means one set is built
rather than one set each.

The ``versa`` import stays inside the cached functions so that the existing
error message is still raised at the point of use when versa is missing, rather
than at import time.
"""

from functools import lru_cache


def _import_versa_module(module_name):
    """Import one versa submodule, keeping the error message the helpers print.

    The setup functions are not re-exported at versa top level, so each one has
    to be reached through the module that defines it.
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except Exception as e:
        print("Error: Versa is not properly installed.")
        raise e


@lru_cache(maxsize=None)
def espnet_wer_args():
    """Return the cached ESPnet ASR system used for WER and CER."""
    return _import_versa_module("versa.corpus_metrics.espnet_wer").espnet_wer_setup(
        model_tag="default",
        beam_size=1,
        text_cleaner="whisper_en",
        use_gpu=True,
    )


@lru_cache(maxsize=None)
def owsm_wer_args():
    """Return the cached OWSM ASR system used for WER and CER."""
    return _import_versa_module("versa.corpus_metrics.owsm_wer").owsm_wer_setup(
        model_tag="default",
        beam_size=1,
        text_cleaner="whisper_en",
        use_gpu=True,
    )


@lru_cache(maxsize=None)
def whisper_wer_args():
    """Return the cached Whisper ASR system used for WER and CER."""
    return _import_versa_module("versa.corpus_metrics.whisper_wer").whisper_wer_setup(
        model_tag="default",
        beam_size=1,
        text_cleaner="whisper_en",
        use_gpu=True,
    )


@lru_cache(maxsize=None)
def pseudo_mos_args():
    """Return the cached (predictor_dict, predictor_fs) for utmos/dnsmos/plcmos."""
    return _import_versa_module("versa.utterance_metrics.pseudo_mos").pseudo_mos_setup(
        use_gpu=True,
        predictor_types=["utmos", "dnsmos", "plcmos"],
        predictor_args={
            "utmos": {"fs": 16000},
            "dnsmos": {"fs": 16000},
            "plcmos": {"fs": 16000},
        },
    )


@lru_cache(maxsize=None)
def sheet_ssqa_model():
    """Return the cached SHEET SSQA model."""
    return _import_versa_module("versa.utterance_metrics.sheet_ssqa").sheet_ssqa_setup(
        model_tag="default",
        model_path=None,
        model_config=None,
        use_gpu=True,
    )
