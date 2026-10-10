"""The OWSM system's side of :mod:`espnet3.api.inference`.

The model behind an ``s2t`` tag is
:class:`espnet2.bin.s2t_inference.Speech2Text` - OWSM, OWSM-CTC and POWSM
checkpoints - whose ``decode_long`` takes one recording of any length and
returns ``(start, end, text)`` per utterance, cutting the recording up the
way that checkpoint was trained to be read. This adapter is the whole
distance between that and the contract: the model's prompt symbols
(language, task, the text said before) as optional inputs, the utterances
as ``segments`` and their texts joined as ``text``, each utterance yielded
as soon as it is decoded. Building the ``Speech2Text``, loading a bundle
and the rate come from
:class:`espnet3.systems.base.backend_inference.BackendInference`.

Examples:
    A published OWSM checkpoint, which has no bundle and so no ``meta.yaml``
    to name its system::

        >>> from espnet3.api.inference import load
        >>> model = load("espnet/owsm_v4_base_102M", system="s2t")
        >>> model("talk.wav")
        {'text': 'hello world ...', 'segments': [{'text': 'hello world',
         'start': 0.0, 'end': 2.4}, ...]}
        >>> model("talk.wav", language="jpn")              # ISO 639-3
        >>> model("talk.wav", language="deu", task="st_eng")  # translate

    Each utterance as it is decoded, rather than all of them at the end::

        >>> for piece in model.stream([{"speech": "talk.wav"}]):
        ...     print(piece["text"], end="", flush=True)

    Any ``Speech2Text`` argument replaces the checkpoint's, as ESPnet2's
    ``from_pretrained`` takes it::

        >>> load("espnet/owsm_v4_base_102M", system="s2t", beam_size=5)

    In the ``infer`` stage, ``inference.yaml`` names this class where it
    named ``Speech2Text``, with the same arguments::

        model:
          _target_: espnet3.systems.esp2_s2t.inference.Inference
          s2t_train_config: ${exp_dir}/config.yaml
          s2t_model_file: ${exp_dir}/valid.acc.ave.pth
          beam_size: 5
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Optional

from espnet3.api.inference import Audio, Field
from espnet3.api.inference.base import gather
from espnet3.systems.base.backend_inference import BackendInference


class Inference(BackendInference):
    """Transcribe or translate with an OWSM model.

    Built three ways, all ending in ``self.backend`` (see
    :class:`BackendInference`): ``Inference.from_pretrained(tag_or_dir)``
    from a ``pack_model`` bundle or a published checkpoint;
    ``Inference(s2t_train_config=..., s2t_model_file=..., beam_size=...)``
    from ``Speech2Text``'s own arguments; ``Inference(speech2text)`` around
    one already built.

    Called with one recording of any length it returns ``{"text": str,
    "segments": [...]}``: the model's utterances with their start and end
    in seconds, and their texts joined by spaces. The recording is read
    through the backend's ``decode_long``, so a CTC-only checkpoint is read
    in overlapping buffers and gives one segment, an encoder-decoder one
    utterance by utterance on its own timestamps.

    The optional inputs are the prompt symbols the model takes, named
    without their angle brackets and checked against the checkpoint's
    vocabulary before anything is decoded:

    - ``language``: ISO 639-3, such as ``"eng"`` or ``"jpn"``. Left out,
      the backend's own ``lang_sym`` when the checkpoint has it, else the
      checkpoint's symbol for working the language out itself
      (``no_language()``: ``<nolang>`` for OWSM, ``<unk>`` for POWSM).
    - ``task``: ``"asr"``, or ``"st_eng"`` and the like for translation.
      Left out, the backend's own ``task_sym``, ``<asr>`` unless built
      otherwise.
    - ``previous_text``: what was said just before the recording, given to
      the model as its first condition; each later utterance is then also
      conditioned on the text decoded before it (``condition_on_prev_text``).

    Examples:
        >>> model = Inference.from_pretrained("espnet/owsm_v4_base_102M")
        >>> model("talk.wav")["text"]
        >>> model("talk.wav", language="jpn", task="st_eng")["segments"]
        >>> model.sample_rate       # the rate the checkpoint was trained at
        16000
    """

    backend_class = "espnet2.bin.s2t_inference.Speech2Text"
    inputs = (
        Field("speech", "audio", "Speech"),
        Field("language", "text", "Language", optional=True),
        Field("task", "text", "Task", optional=True),
        Field("previous_text", "text", "Previous text", optional=True),
    )
    outputs = (
        Field("text", "text", "Transcription"),
        Field("segments", "segments", "Segments"),
    )

    def _known(self, symbol: str) -> bool:
        """Whether the checkpoint's vocabulary has ``symbol``.

        A model that does not list its tokens is given the benefit of the
        doubt, as the backend's ``no_language()`` gives it.
        """
        tokens = getattr(getattr(self.backend, "s2t_model", None), "token_list", None)
        return not tokens or symbol in tokens

    def _language(self, language: Optional[str]) -> str:
        """Return the language symbol to decode with.

        Raises:
            ValueError: If ``language`` is not in the checkpoint's
                vocabulary, or none was given and the checkpoint has no
                symbol for working the language out itself.
        """
        if language is not None:
            symbol = f"<{language}>"
            if not self._known(symbol):
                raise ValueError(
                    f"language {language!r}: this model has no {symbol}; pass "
                    "ISO 639-3, three letters: eng, deu, jpn, zho, fra, spa ..."
                )
            return symbol
        own = getattr(self.backend, "lang_sym", None)
        if own and self._known(own):
            return own
        try:
            return self.backend.no_language()
        except ValueError as e:
            raise ValueError(f"{e}; pass language=<ISO 639-3>") from e

    def _task(self, task: Optional[str]) -> Optional[str]:
        """Return the task symbol to decode with, or ``None`` for the backend's own.

        Raises:
            ValueError: If ``task`` is not in the checkpoint's vocabulary.
        """
        if task is None:
            return None
        symbol = f"<{task}>"
        if not self._known(symbol):
            raise ValueError(
                f"task {task!r}: this model has no {symbol}; it takes asr, "
                "or st_<ISO 639-3> to translate into that language"
            )
        return symbol

    def run_stream(
        self, chunks: Iterable[Mapping[str, Any]]
    ) -> Iterator[Mapping[str, Any]]:
        """Yield each utterance as the model decodes it.

        The model reads the whole recording to cut it up, so the input is
        gathered first; what streams is the output, one chunk per utterance
        as ``decode_long`` would list it. A text piece is what is new: the
        utterance's text, with the space that joins it to the one before.
        ``run`` is the default, which gathers these.

        Yields:
            ``{"text": piece, "segments": [one segment]}`` per utterance,
            and ``{"text": "", "segments": []}`` for a recording the model
            found nothing in.
        """
        inputs = gather(self.inputs, chunks)
        speech: Audio = inputs["speech"]
        kwargs: dict[str, Any] = {
            "lang_sym": self._language(inputs.get("language")),
            "task_sym": self._task(inputs.get("task")),
        }
        previous = inputs.get("previous_text")
        if previous is not None:
            kwargs.update(init_text=previous, condition_on_prev_text=True)
        first = True
        for start, end, text in self.backend.iter_long(speech.array, **kwargs):
            yield {
                "text": text if first else " " + text,
                "segments": [{"text": text, "start": start, "end": end}],
            }
            first = False
        if first:
            yield {"text": "", "segments": []}
