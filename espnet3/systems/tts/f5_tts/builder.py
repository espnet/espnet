"""Build the F5-TTS model for ESPnet3's Hydra instantiation path.

Replaces the ``espnet2.tasks.tts.TTSTask.build_model`` route: F5-TTS is an
ESPnet3 model and is no longer registered in ``espnet2/tasks/tts.py``, so the
training config reaches it through ``_target_`` instead of ``task:``. Only the
branches this recipe exercises are reproduced, i.e. mel extracted inside the
model, no normalization layer, and no pitch/energy predictors.

The returned object is still ``espnet2.tts.espnet_model.ESPnetTTSModel``, which
keeps the forward maths and the ``collect_feats`` contract identical to the
espnet2 task route.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from espnet2.tts.espnet_model import ESPnetTTSModel
from espnet3.systems.tts.f5_tts.f5tts import F5TTS
from espnet3.systems.tts.f5_tts.vocoder_mel import VocoderMelSpec


def _plain(value: Any) -> Any:
    """Return ``value`` with any OmegaConf container turned into plain Python.

    Hydra hands nested blocks over as ``DictConfig``/``ListConfig`` unless the
    config opts into ``_convert_``. Those unpack through ``**`` well enough, but
    they leak into ``F5TTS``'s stored attributes and into checkpointed hparams,
    so they are converted once here instead.
    """
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def build_f5_tts_model(
    token_list: Union[str, Sequence[str]],
    feats_extract_conf: Dict[str, Any],
    tts_conf: Dict[str, Any],
    model_conf: Optional[Dict[str, Any]] = None,
    odim: Optional[int] = None,
) -> ESPnetTTSModel:
    """Assemble ``ESPnetTTSModel(feats_extract=VocoderMelSpec, tts=F5TTS)``.

    Every hyper-parameter reaches its component through one of the three
    ``*_conf`` blocks, whose keys map one-to-one onto that component's
    ``__init__``. None of them accepts ``**kwargs``, so a misspelled key raises
    ``TypeError`` at build time rather than silently training a default model:

    - ``tts_conf`` -> :class:`F5TTS` (model size and flow-matching settings)
    - ``feats_extract_conf`` -> :class:`VocoderMelSpec` (mel front-end)
    - ``model_conf`` -> :class:`ESPnetTTSModel` (wrapper-level settings)

    To scale the model, set the :class:`F5TTS` backbone sizes in ``tts_conf``.
    They default to F5TTS_Base (``dim: 1024``, ``depth: 22``, ``heads: 16``);
    the example below is F5TTS_Small.

    Args:
        token_list: Path to the token file, or the token list itself. Its
            length becomes the vocabulary size, i.e. ``F5TTS(idim=...)``.
        feats_extract_conf: Keyword arguments for ``VocoderMelSpec``.
        tts_conf: Keyword arguments for ``F5TTS`` beyond ``idim``/``odim``.
        model_conf: Extra keyword arguments for ``ESPnetTTSModel``.
        odim: Must stay ``None``. F5-TTS extracts mel inside the model, so the
            output dimension comes from the feature extractor.

    Returns:
        The assembled TTS model.

    Raises:
        RuntimeError: If ``token_list`` is neither a path nor a sequence, or if
            ``odim`` is given explicitly.
        TypeError: If an unknown key is present in the ``model:`` block or in
            any of the ``*_conf`` blocks.

    Example:
        .. code-block:: yaml

            model:
              _target_: espnet3.systems.tts.f5_tts.builder.build_f5_tts_model
              token_list: ${data_dir}/tokens/char_tokens.txt
              feats_extract_conf:
                fs: 24000
                n_fft: 1024
                hop_length: 256
                win_length: 1024
                n_mels: 100
                mel_spec_type: vocos
              tts_conf:          # F5TTS_Small; omit these keys for F5TTS_Base
                dim: 768
                depth: 18
                heads: 12
                dim_head: 64
                ff_mult: 2
                text_dim: 512
                conv_layers: 4
                odeint_method: euler
              model_conf: {}
    """
    token_list = _plain(token_list)
    if isinstance(token_list, str):
        with open(token_list, encoding="utf-8") as f:
            tokens: List[str] = [line[0] + line[1:].rstrip() for line in f]
    elif isinstance(token_list, (tuple, list)):
        tokens = list(token_list)
    else:
        raise RuntimeError("token_list must be a path or a sequence of tokens")

    vocab_size = len(tokens)
    logging.info(f"Vocabulary size: {vocab_size}")

    if odim is not None:
        raise RuntimeError(
            "F5-TTS extracts mel inside the model, so `odim` must stay null "
            "and is taken from VocoderMelSpec.output_size()."
        )

    feats_extract = VocoderMelSpec(**_plain(feats_extract_conf))
    tts = F5TTS(
        idim=vocab_size,
        odim=feats_extract.output_size(),
        **_plain(tts_conf),
    )
    # ``ESPnetTTSModel`` declares these without defaults, so they are passed
    # explicitly as None, exactly as ``TTSTask.build_model`` does for a config
    # with no normalization and no pitch/energy predictors.
    return ESPnetTTSModel(
        feats_extract=feats_extract,
        pitch_extract=None,
        energy_extract=None,
        normalize=None,
        pitch_normalize=None,
        energy_normalize=None,
        tts=tts,
        **(_plain(model_conf) or {}),
    )
