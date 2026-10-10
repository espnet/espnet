"""The ESPnet2 ASR system's side of :mod:`espnet3.api.inference`.

The model behind an ``asr`` bundle is
:class:`espnet2.bin.asr_inference.Speech2Text`, which takes one waveform and
returns an n-best list of ``(text, tokens, token_ids, hypothesis)``. This
adapter is the whole distance between that and the contract: the best
text, under the key every transcriber uses. Building the ``Speech2Text``,
loading a bundle and the frontend's rate come from
:class:`espnet3.systems.base.backend_inference.BackendInference`.

Examples:
    From a bundle, by directory or Hub tag::

        >>> from espnet3.systems.esp2_asr.inference import Inference
        >>> model = Inference.from_pretrained("exp/train/model_pack")
        >>> model("utt.wav")
        {'text': 'hello world'}

    Or, without naming the system, through the bundle's ``meta.yaml``::

        >>> from espnet3.api.inference import load
        >>> load("espnet/some_asr_pack", device="cuda:0")("utt.wav")["text"]

    An ESPnet2 ASR model published without a bundle has no ``meta.yaml``
    to name its system, so the caller does; ``Speech2Text.from_pretrained``
    reads it, and the transducer ``Speech2Text`` is not reached this way::

        >>> load("espnet/some_espnet2_asr_model", system="asr")

    Any ``Speech2Text`` argument replaces the packed one, as ESPnet2's
    ``from_pretrained`` takes it - decoding settings or anything else::

        >>> load("espnet/some_asr_pack", beam_size=5, ctc_weight=0.3, nbest=3)

    In the ``infer`` stage, ``inference.yaml`` names this class where it
    named ``Speech2Text``, with the same arguments; for a transducer, name
    that class as ``backend_class``::

        model:
          _target_: espnet3.systems.esp2_asr.inference.Inference
          asr_train_config: ${exp_dir}/config.yaml
          asr_model_file: ${exp_dir}/valid.acc.ave.pth
          beam_size: 10
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from espnet3.api.inference import Audio, Field
from espnet3.systems.base.backend_inference import BackendInference, parse_rate

# espnet2.asr.frontend.default.DefaultFrontend(fs=16000): what an ESPnet2
# model runs at when its training config says nothing about the rate.
DEFAULT_FRONTEND_RATE = 16000


class Inference(BackendInference):
    """Transcribe with an ESPnet2 ASR model.

    Built three ways, all ending in ``self.backend`` (see
    :class:`BackendInference`):

    - ``Inference.from_pretrained(tag_or_dir, device=...)`` from a
      ``pack_model`` bundle, or an ESPnet2 ASR model on the Hub;
    - ``Inference(asr_train_config=..., asr_model_file=..., beam_size=...)``
      from ``Speech2Text``'s own arguments, which is what ``inference.yaml``
      does, plus ``backend_class=`` for the transducer ``Speech2Text``;
    - ``Inference(speech2text)`` around one already built.

    Called with one utterance (a path, a ``(rate, samples)`` pair, an array
    or an :class:`~espnet3.api.inference.Audio`) it returns ``{"text": str}``;
    ``model.batch(items)`` decodes several in one beam search.

    A multichannel recording reaches the model as ESPnet2's own
    ``Speech2Text`` would take it: every channel, ``(samples, channels)``,
    when the model uses them (:attr:`takes_channels`), else channel 0 -
    the one ``DefaultFrontend`` picks at inference, and the only input an
    s3prl, Whisper or fused frontend takes.

    Examples:
        >>> model = Inference.from_pretrained("espnet/some_asr_pack")
        >>> model("utt.wav")
        {'text': 'hello world'}
        >>> model.batch([{"speech": "a.wav"}, {"speech": "b.wav"}])
        [{'text': '...'}, {'text': '...'}]
        >>> model.sample_rate       # the frontend's rate, read off the config
        16000
    """

    backend_class = "espnet2.bin.asr_inference.Speech2Text"
    # every channel arrives; run() decides what the model gets (takes_channels)
    inputs = (Field("speech", "audio", "Speech", channels=None),)
    outputs = (Field("text", "text", "Transcription"),)

    def __init__(
        self,
        backend: Any = None,
        *,
        device: str = "cpu",
        backend_class: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Build or keep the backend, asking a transducer for decoded text.

        The transducer ``Speech2Text`` returns raw hypotheses unless built
        with ``return_decoded_hyp=True``, and this class reads the text, so
        it builds one that way; ``return_decoded_hyp=False`` is refused
        rather than failing at the first utterance. Everything else is
        :class:`BackendInference`'s.

        Raises:
            ValueError: If ``return_decoded_hyp`` is given as false for a
                backend that takes it.
        """
        if backend is None:
            built = self._backend_type(backend_class)
            if "return_decoded_hyp" in inspect.signature(built).parameters:
                if not kwargs.get("return_decoded_hyp", True):
                    raise ValueError(
                        "return_decoded_hyp=False: this class reads the decoded "
                        "text, so the transducer Speech2Text must return it"
                    )
                kwargs["return_decoded_hyp"] = True
        super().__init__(backend, device=device, backend_class=backend_class, **kwargs)

    @property
    def takes_channels(self) -> bool:
        """Whether the model uses every channel of a multichannel recording.

        True for a joint enhancement-and-ASR model, and for a
        ``DefaultFrontend`` whose enhancement stage runs WPE or a
        beamformer before it picks a channel: ESPnet2's ``Speech2Text``
        gets every channel there, so this class passes them. False
        otherwise, and channel 0 is passed. That is no loss: the stage
        ``DefaultFrontend`` builds by default enables neither, lets the
        channels through, and channel 0 is then what it picks.
        """
        if getattr(self.backend, "enh_s2t_task", False):
            return True
        frontend = getattr(getattr(self.backend, "asr_model", None), "frontend", None)
        enhancer = getattr(frontend, "frontend", None)
        return bool(
            getattr(enhancer, "use_wpe", False)
            or getattr(enhancer, "use_beamformer", False)
        )

    def _waveform(self, speech: Audio) -> np.ndarray:
        """Return the samples the backend takes, in ESPnet2's channel order."""
        if speech.channels > 1 and self.takes_channels:
            return speech.array.T  # ESPnet2 takes (samples, channels)
        return speech.mono().array

    @property
    def sample_rate(self) -> int:
        """The frontend's rate: ``frontend_conf.fs``, else ESPnet2's 16 kHz.

        An ESPnet2 training config may leave ``frontend_conf`` empty, as the
        mini_an4 recipe does; the frontend it builds then runs at
        :class:`~espnet2.asr.frontend.default.DefaultFrontend`'s own default
        of 16 kHz, so that is the rate, not a guess. A backend that is not a
        ``Speech2Text`` (no ``asr_train_args``) falls back to the base
        class, which raises.
        """
        args = getattr(self.backend, "asr_train_args", None)
        if args is None:
            return (
                super().sample_rate
            )  # not a Speech2Text: its own attribute, or an error
        conf = getattr(args, "frontend_conf", None) or {}
        return parse_rate(conf.get("fs", DEFAULT_FRONTEND_RATE))

    def run(self, speech: Audio) -> Mapping[str, Any]:
        """Return the best hypothesis' text.

        Args:
            speech: The utterance, at :attr:`sample_rate`, every channel.

        Returns:
            ``{"text": str}``. For a joint enhancement-and-ASR model, which
            returns one n-best list per speaker, the first speaker's.
        """
        return _text(self.backend(self._waveform(speech)))

    def run_batch(self, items: Sequence[Mapping[str, Any]]) -> Sequence[Mapping]:
        """Decode a batch in one beam search when the backend can.

        ``espnet2.bin.asr_inference.Speech2Text`` decodes a list of
        waveforms together (``batch_decode``); a backend without it, such
        as the transducer ``Speech2Text``, gets the items one by one, and
        so does a batch with several channels for a model that takes them,
        since ``batch_decode`` reads each utterance as one channel.
        """
        if len(items) < 2 or not callable(getattr(self.backend, "batch_decode", None)):
            return super().run_batch(items)
        waves = [self._waveform(item["speech"]) for item in items]
        if any(wave.ndim > 1 for wave in waves):
            return super().run_batch(items)
        results = self.backend(waves)
        return [_text(nbest) for nbest in results]


def _text(nbest: Any) -> dict:
    """Return the best hypothesis' text from an n-best list."""
    best = nbest[0]
    # Speech2Text returns (text, tokens, token_ids, hyp); a joint enh+ASR
    # model returns one such list per speaker, of which this is the first.
    if isinstance(best, list):
        best = best[0]
    if not isinstance(best, tuple):
        raise TypeError(
            f"the backend returned a {type(best).__name__}, not a decoded "
            "(text, tokens, token_ids, hyp) tuple; a transducer Speech2Text "
            "built elsewhere needs return_decoded_hyp=True"
        )
    return {"text": best[0]}
