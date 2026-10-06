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

    In the ``infer`` stage, ``inference.yaml`` names this class where it
    named ``Speech2Text``, with the same arguments; for a transducer, name
    that class as ``backend_class`` and set ``return_decoded_hyp: true``::

        model:
          _target_: espnet3.systems.esp2_asr.inference.Inference
          asr_train_config: ${exp_dir}/config.yaml
          asr_model_file: ${exp_dir}/valid.acc.ave.pth
          beam_size: 10
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from espnet3.components.contract import Audio, Field
from espnet3.systems.base.backend_inference import BackendInference, parse_rate

# espnet2.asr.frontend.default.DefaultFrontend(fs=16000): what an ESPnet2
# model runs at when its training config says nothing about the rate.
DEFAULT_FRONTEND_RATE = 16000


class Inference(BackendInference):
    """Transcribe with an ESPnet2 ASR model.

    Built three ways, all ending in ``self.backend`` (see
    :class:`BackendInference`):

    - ``Inference.from_pretrained(tag_or_dir, device=...)`` from a
      ``pack_model`` bundle or Hub tag;
    - ``Inference(asr_train_config=..., asr_model_file=..., beam_size=...)``
      from ``Speech2Text``'s own arguments, which is what ``inference.yaml``
      does, plus ``backend_class=`` for the transducer ``Speech2Text``;
    - ``Inference(speech2text)`` around one already built.

    Called with one utterance (a path, a ``(rate, samples)`` pair, an array
    or an :class:`~espnet3.components.contract.Audio`) it returns
    ``{"text": str}``; ``model.batch(items)`` decodes several in one beam
    search.

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
    inputs = (Field("speech", "audio", "Speech"),)
    outputs = (Field("text", "text", "Transcription"),)

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
            speech: The utterance, at :attr:`sample_rate`.

        Returns:
            ``{"text": str}``. For a joint enhancement-and-ASR model, which
            returns one n-best list per speaker, the first speaker's.
        """
        return _text(self.backend(speech.array))

    def run_batch(self, items: Sequence[Mapping[str, Any]]) -> Sequence[Mapping]:
        """Decode a batch in one beam search when the backend can.

        ``espnet2.bin.asr_inference.Speech2Text`` decodes a list of
        waveforms together (``batch_decode``), 1.5-2.5x faster on a GPU
        than one at a time; a backend without it, such as the transducer
        ``Speech2Text``, gets the items one by one.
        """
        if len(items) < 2 or not callable(getattr(self.backend, "batch_decode", None)):
            return super().run_batch(items)
        results = self.backend([item["speech"].array for item in items])
        return [_text(nbest) for nbest in results]


def _text(nbest: Any) -> dict:
    """Return the best hypothesis' text from an n-best list."""
    best = nbest[0]
    # Speech2Text returns (text, tokens, token_ids, hyp); a joint enh+ASR
    # model returns one such list per speaker, of which this is the first.
    if isinstance(best, list):
        best = best[0]
    return {"text": best[0]}
