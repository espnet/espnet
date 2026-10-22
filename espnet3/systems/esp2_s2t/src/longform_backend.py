"""Long-form decoding for ESPnet2's ``s2t`` and ``s2t_ctc`` tasks.

The backend ``conf/inference.yaml`` names as ``model._target_``. It owns
everything particular to an OWSM-style checkpoint -- resolving the language
and task symbols against its vocabulary, cutting a recording into utterances,
and the two cases where that cannot be done in one pass -- and returns
``(start, end, text)`` tuples, which
:class:`~espnet3.systems.esp2_s2t.inference.ESP2S2TInferenceRunner` shapes
into the contracted ``text`` and ``segments``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
)

logger = logging.getLogger(__name__)

Utterance = tuple


class S2TLongFormBackend:
    """Decode a recording of any length with an OWSM-style checkpoint.

    Args:
        model: A constructed ``espnet2.bin.s2t_inference.Speech2Text``.
        **unused: Ignored, so a config may carry keys this does not read.

    Examples:
        >>> backend = S2TLongFormBackend(model=speech2text)  # doctest: +SKIP
        >>> backend(speech=array)                            # doctest: +SKIP
        [(0.0, 4.2, 'hello there')]
    """

    def __init__(self, model: Any, **unused: Any) -> None:
        self.model = model

    @property
    def sample_rate(self) -> int:
        """The rate the checkpoint was trained at."""
        conf = getattr(self.model, "preprocessor_conf", None) or {}
        return int(conf.get("fs", 16000))

    # -- vocabulary -------------------------------------------------------

    def _known(self, symbol: str) -> bool:
        """Whether the checkpoint's vocabulary has ``symbol``.

        A model that does not list its tokens is given the benefit of the
        doubt, as the backend's ``no_language()`` gives it.
        """
        tokens = getattr(getattr(self.model, "s2t_model", None), "token_list", None)
        return not tokens or symbol in tokens

    def _targets(self) -> list:
        """Return the languages the checkpoint translates into, by its <st_xxx>."""
        tokens = getattr(getattr(self.model, "s2t_model", None), "token_list", None)
        return [t[len("<st_") : -1] for t in tokens or () if t.startswith("<st_")]

    def language_symbol(self, language: Optional[str]) -> str:
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
        try:
            return self.model.no_language()
        except ValueError as e:
            raise ValueError(f"{e}; pass language=<ISO 639-3>") from e

    def task_symbol(self, task: Optional[str]) -> Optional[str]:
        """Return the task symbol, or ``None`` for the checkpoint's own.

        Raises:
            ValueError: If ``task`` is not in the checkpoint's vocabulary.
        """
        if task is None:
            return None
        symbol = f"<{task}>"
        if not self._known(symbol):
            # the translation targets are few, one <st_xxx> each, so they are
            # named: `kor` is ISO 639-3, and "use ISO 639-3" would not help
            targets = ", ".join(sorted(self._targets())) or "nothing"
            raise ValueError(
                f"task {task!r}: this model has no {symbol}; it takes asr, or "
                f"st_<ISO 639-3> to translate, into {targets}"
            )
        return symbol

    # -- the two cases long-form decoding cannot cover --------------------

    def phones_by_window(self, task_sym: Optional[str]) -> bool:
        """Whether this task on this checkpoint is decoded a window at a time.

        Phone recognition has no timestamps to cut a recording up by, and an
        encoder-decoder checkpoint asked for phones over the silence a short
        window is padded with repeats what it has already said, for as long
        as the window lasts; so each window is decoded on its own, as the
        POWSM model card does. A CTC-only checkpoint has neither problem and
        takes the long-form path, as every other task does.
        """
        task_sym = task_sym or getattr(self.model, "task_sym", None)
        return task_sym == "<pr>" and not getattr(self.model, "ctc_only", False)

    def _phones_window_by_window(
        self, array: np.ndarray, lang_sym: str, previous_text: Optional[str] = None
    ) -> Iterator[Utterance]:
        """Yield phones for each window of the model's length, one segment each.

        ``previous_text`` conditions the first window, as ``decode_long``'s
        ``init_text`` conditions the first segment.
        """
        conf = self.model.preprocessor_conf
        window = int(conf["fs"] * conf["speech_length"])
        rate = self.sample_rate
        for start in range(0, max(len(array), 1), window):
            results = self.model(
                array[start : start + window],
                text_prev=previous_text if start == 0 else None,
                lang_sym=lang_sym,
                task_sym="<pr>",
            )
            text = results[0][3] if results else ""  # without the special tokens
            end = min(start + window, len(array))
            yield (start / rate, end / rate, text)

    def batchable_samples(self) -> Optional[int]:
        """Return the longest recording a batch item may be, or ``None`` if unsaid.

        One window of the model (``speech_length`` seconds at ``fs``) less the
        second before its end, where ``decode_long`` reads an utterance as cut
        off.
        """
        conf = getattr(self.model, "preprocessor_conf", None) or {}
        if "fs" not in conf or "speech_length" not in conf:
            return None
        return int(conf["fs"] * (conf["speech_length"] - 1))

    # -- decoding ---------------------------------------------------------

    def decode(
        self,
        speech: np.ndarray,
        language: Optional[str] = None,
        task: Optional[str] = None,
        previous_text: Optional[str] = None,
    ) -> list:
        """Return ``(start, end, text)`` for every utterance in one recording.

        A recording the model finds nothing in returns ``[]``, including one
        too short for the encoder to subsample: espnet2 raises
        ``TooShortUttError`` for those, and an empty hypothesis is still
        scored against the reference, where raising would drop the utterance
        and flatter the model.
        """
        lang_sym = self.language_symbol(language)
        task_sym = self.task_symbol(task)
        kwargs: dict = {"lang_sym": lang_sym, "task_sym": task_sym}
        if previous_text is not None:
            kwargs.update(init_text=previous_text, condition_on_prev_text=True)

        if self.phones_by_window(task_sym):
            source: Any = self._phones_window_by_window(speech, lang_sym, previous_text)
        else:
            source = self.model.iter_long(speech, **kwargs)

        out: list = []
        try:
            for utterance in source:
                out.append(utterance)
        except TooShortUttError as e:
            logger.warning("%s: emitting an empty hypothesis", e)
        return out

    def decode_batch(self, items: Sequence[dict]) -> list:
        """Decode several recordings, grouping the ones one window can hold.

        ``Speech2Text.batch_decode`` decodes a minibatch of single-window
        utterances together, with one language and task symbol for the batch;
        so the items that end at least a second before the model's window does
        and carry no ``previous_text`` are grouped by their symbols and decoded
        a group at a time, with ``predict_time`` so their segments are read off
        the timestamps exactly as ``decode_long`` reads a window. Everything
        else -- a longer recording, an item with ``previous_text``, phones on
        an encoder-decoder checkpoint, a group of one, and any checkpoint
        without ``batch_decode`` or whose config does not say its window --
        goes one at a time.
        """
        batchable = self.batchable_samples()
        outputs: list = [None] * len(items)
        groups: dict = {}

        for i, item in enumerate(items):
            key = (
                self.language_symbol(item.get("language")),
                self.task_symbol(item.get("task")),
            )
            if (
                batchable is None
                or not hasattr(self.model, "batch_decode")
                or item.get("previous_text") is not None
                or len(item["speech"]) > batchable
                or self.phones_by_window(key[1])
            ):
                outputs[i] = self.decode(**item)
                continue
            groups.setdefault(key, []).append(i)

        for (lang_sym, task_sym), index in groups.items():
            if len(index) == 1:
                outputs[index[0]] = self.decode(**items[index[0]])
                continue
            try:
                results = self.model.batch_decode(
                    [items[i]["speech"] for i in index],
                    lang_sym=lang_sym,
                    task_sym=task_sym,
                    predict_time=True,
                )
            except TooShortUttError:
                # The search is over the whole group, so one clip too short
                # to subsample fails all of them; they are decoded one by
                # one instead, where only that clip comes back empty.
                for i in index:
                    outputs[i] = self.decode(**items[i])
                continue
            if len(results) != len(index):
                raise RuntimeError(
                    f"batch_decode returned {len(results)} results for "
                    f"{len(index)} utterances"
                )
            for i, nbest in zip(index, results):
                outputs[i] = list(self.model.window_utterances(nbest[0][2]))
        return outputs

    def __call__(self, speech, **kwargs):
        """Decode one recording, or a list of them.

        Args:
            speech: One waveform, or a list of them for a batch.
            **kwargs: ``language``, ``task`` and ``previous_text``, each one
                value or a list as long as ``speech``.

        Returns:
            ``(start, end, text)`` tuples for one recording, or a list of
            those lists for a batch.
        """
        if not isinstance(speech, list):
            return self.decode(speech, **kwargs)

        def at(key, i):
            value = kwargs.get(key)
            return value[i] if isinstance(value, list) else value

        items = [
            {
                "speech": array,
                "language": at("language", i),
                "task": at("task", i),
                "previous_text": at("previous_text", i),
            }
            for i, array in enumerate(speech)
        ]
        return self.decode_batch(items)
