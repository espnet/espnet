"""S2T inference with the serialized-output-training timestamp rules.

A subclass of ``espnet2.bin.s2t_inference.Speech2Text`` rather than a copy, so
the decoder keeps every espnet2 fix. It adds two options. ``speaker_change_symbol``
scopes Whisper's timestamp rules to the current speaker block, and
``adaptive_timestamp`` adds Whisper's last rule, which needs the decoder's
probabilities and so wraps the decoder scorer. Both are installed on the beam
search after the parent has built it.
"""

from __future__ import annotations

import functools
import inspect
from typing import Optional

import torch

from espnet2.bin.s2t_inference import Speech2Text as _Speech2Text
from espnet2.bin.s2t_inference import _is_ctc_only
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.scorer_interface import BatchScorerInterface
from espnet3.systems.esp2_s2t.whisper_timestamp_scorer import (
    WhisperTimestampFilter,
    wrap_timestamp_decoder,
)

_PARENT_SIGNATURE = inspect.signature(_Speech2Text.__init__)


def _primer_length(primer) -> int:
    """Return the number of prompt tokens for a timestamp filter.

    ``hyp_primer`` is one sequence for the full batch, or one sequence for
    each utterance. Beam search moves all hypotheses together. All sequences
    must therefore have the same length.
    """
    if (
        primer is not None
        and len(primer) > 0
        and isinstance(primer[0], (list, tuple, torch.Tensor))
    ):
        lengths = {len(p) for p in primer}
        if len(lengths) != 1:
            raise ValueError(
                "the timestamp filter needs a single prompt length for the "
                f"batch, got {sorted(lengths)}. Decode these utterances with "
                "--batch_size 1."
            )
        return lengths.pop()
    return len(primer)


def _set_hyp_primer_and_sample_begin(beam_search, timestamp_filter, primer):
    # Every decoding path sets its prompt through set_hyp_primer, so the
    # filter learns where generated tokens start whenever the prompt changes.
    # A module-level function under functools.partial, not a closure, so the
    # beam search stays picklable.
    type(beam_search).set_hyp_primer(beam_search, primer)
    if primer is not None:
        timestamp_filter.set_sample_begin(_primer_length(primer))


class Speech2Text(_Speech2Text):
    """espnet2's S2T decoder with the serialized-output-training timestamp rules.

    Args:
        speaker_change_symbol: The separator the checkpoint was trained with.
            When given, the timestamp rules apply to the current speaker block
            only, so a new speaker may open anywhere in the window. It must be
            a row of the checkpoint's token list.
        adaptive_timestamp: Add Whisper's last rule, which forces a timestamp
            when the summed timestamp probability beats the best text token.
        *args: As for ``espnet2.bin.s2t_inference.Speech2Text``.
        **kwargs: As for ``espnet2.bin.s2t_inference.Speech2Text``.
    """

    def __init__(
        self,
        *args,
        speaker_change_symbol: Optional[str] = None,
        adaptive_timestamp: bool = False,
        **kwargs,
    ):
        """Build the parent decoder, then install the timestamp rules."""
        given = _PARENT_SIGNATURE.bind(self, *args, **kwargs)
        given.apply_defaults()
        requested = [
            name
            for name, on in (
                ("speaker_change_symbol", speaker_change_symbol is not None),
                ("adaptive_timestamp", adaptive_timestamp),
            )
            if on
        ]
        if requested and _is_ctc_only(given.arguments["s2t_train_config"]):
            # Refused before the model is built: building the filter reads
            # timestamp symbols a CTC-only preprocessor_conf need not carry,
            # and a CTC-only beam search would drop the filter without a word.
            raise ValueError(
                f"{' and '.join(requested)} needs a decoder with timestamp "
                "tokens, and this checkpoint is CTC-only"
            )
        if adaptive_timestamp and given.arguments["ctc_weight"] == 1.0:
            # The adaptive rule rides on the decoder score, whose weight is
            # 1 - ctc_weight. At zero BeamSearch drops the scorer and the
            # timestamp rules with it. speaker_change_symbol on its own is
            # unaffected: it replaces scorefilter, at a fixed weight.
            raise ValueError(
                "adaptive_timestamp carries the timestamp rules on the "
                "decoder score, which ctc_weight=1.0 gives no weight"
            )

        super().__init__(*args, **kwargs)

        self.timestamp_filter = None
        if not requested:
            return

        token_list = self.s2t_model.token_list
        conf = self.s2t_train_args.preprocessor_conf
        self.timestamp_filter = WhisperTimestampFilter(
            first_time=token_list.index(conf["first_time_symbol"]),
            last_time=token_list.index(conf["last_time_symbol"]),
            eos=self.s2t_model.eos,
            vocab_size=len(token_list),
            sample_begin=0,  # set again with each new primer
            notimestamps=token_list.index(conf["notime_symbol"]),
            speaker_change=(
                None
                if speaker_change_symbol is None
                else token_list.index(speaker_change_symbol)
            ),
        )
        self._install_timestamp_rules(adaptive_timestamp)

    def _install_timestamp_rules(self, adaptive_timestamp: bool) -> None:
        """Put the filter on the beam search the parent built.

        The parent built the beam search with its own ScoreFilter under
        ``scorefilter``. BeamSearch keeps its scorers in plain dicts, so this
        does by hand what its constructor's loop does: ``scorers`` for lookup,
        ``full_scorers`` for scoring, ``nn_dict`` so ``.to()`` reaches modules.
        """
        beam_search = self.beam_search
        if adaptive_timestamp:
            # The wrapper applies the filter itself, so the filter is not
            # registered twice. The mask now rides on the decoder score,
            # weighted by 1 - ctc_weight; -inf survives any positive weight,
            # and the guard in __init__ refuses a zero one.
            decoder = wrap_timestamp_decoder(
                self.s2t_model.decoder, self.timestamp_filter
            )
            beam_search.scorers["decoder"] = decoder
            beam_search.full_scorers["decoder"] = decoder
            beam_search.nn_dict["decoder"] = decoder
            for table in (beam_search.scorers, beam_search.full_scorers):
                table.pop("scorefilter", None)
            if "scorefilter" in beam_search.nn_dict:
                del beam_search.nn_dict["scorefilter"]
        else:
            beam_search.scorers["scorefilter"] = self.timestamp_filter
            beam_search.full_scorers["scorefilter"] = self.timestamp_filter
            if "scorefilter" in beam_search.nn_dict:
                # The filter is not a module; leaving the parent's would keep
                # an unused ScoreFilter on the device.
                del beam_search.nn_dict["scorefilter"]

        if isinstance(beam_search, BatchBeamSearch):
            # The parent chose BatchBeamSearch from the scorers it had. Ours
            # must keep that choice valid, or batch scoring fails deep inside
            # the search instead of here.
            for name in ("decoder", "scorefilter"):
                scorer = beam_search.full_scorers.get(name)
                if scorer is not None and not isinstance(scorer, BatchScorerInterface):
                    raise TypeError(
                        f"{name} ({type(scorer).__name__}) is not a batch "
                        "scorer, but the beam search is "
                        f"{type(beam_search).__name__}"
                    )

        # The parent's .to() ran before the wrapper existed.
        beam_search.to(device=self.device, dtype=getattr(torch, self.dtype)).eval()
        beam_search.set_hyp_primer = functools.partial(
            _set_hyp_primer_and_sample_begin, beam_search, self.timestamp_filter
        )

    def _build_hyp_primer(
        self, lang_sym=None, task_sym=None, predict_time=None, text_prev=None
    ):
        """Refuse a prompt without timestamps while the timestamp rules are on.

        The filter needs a timestamp as the first new token. A prompt without
        timestamps ends with the no-timestamp symbol, which blocks every
        timestamp, so together they permit no token at all.
        """
        effective = self.predict_time if predict_time is None else predict_time
        if self.timestamp_filter is not None and not effective:
            raise ValueError(
                "speaker_change_symbol and adaptive_timestamp constrain the "
                "timestamp tokens, so they need predict_time=True"
            )
        return super()._build_hyp_primer(lang_sym, task_sym, predict_time, text_prev)
