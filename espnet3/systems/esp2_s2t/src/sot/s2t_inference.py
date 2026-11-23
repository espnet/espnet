"""S2T inference with the timestamp rules for serialized output training.

This module subclasses ``espnet2.bin.s2t_inference.Speech2Text`` instead of
copying it, so fixes in espnet2 also apply here. It adds two options.
``speaker_change_symbol`` applies the Whisper timestamp rules to the current
speaker block only. ``adaptive_timestamp`` adds the last Whisper rule, which
needs the probabilities of the decoder, so it wraps the decoder scorer. Both
are added to the beam search after the parent class builds it. With either
option on, the hypothesis text keeps its timestamps and separators, which the
metrics need.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from espnet2.bin.s2t_inference import Speech2Text as _Speech2Text
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.scorer_interface import BatchScorerInterface
from espnet2.text.whisper_token_id_converter import (
    OpenAIWhisperTokenIDConverter as _DroppingConverter,
)
from espnet3.systems.esp2_s2t.src.sot.whisper_timestamp_scorer import (
    WhisperTimestampFilter,
    wrap_timestamp_decoder,
)
from espnet3.systems.esp2_s2t.src.sot.whisper_token_id_converter import (
    OpenAIWhisperTokenIDConverter,
)


def _primer_length(primer) -> int:
    """Return the number of prompt tokens for a timestamp filter.

    ``hyp_primer`` is one sequence for the whole batch, or one sequence for
    each utterance. Beam search advances all hypotheses together, so all
    sequences must have the same length.
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
    # Every decoding path sets its prompt with set_hyp_primer, so the filter
    # knows where the generated tokens start whenever the prompt changes. This
    # is a module-level function used through functools.partial instead of a
    # closure, so the beam search can still be pickled.
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
        super().__init__(*args, **kwargs)

        self.timestamp_filter = None
        requested = [
            name
            for name, on in (
                ("speaker_change_symbol", speaker_change_symbol is not None),
                ("adaptive_timestamp", adaptive_timestamp),
            )
            if on
        ]
        if not requested:
            return
        if self.ctc_only:
            # A CTC-only model has no decoder for the timestamp rules to act on.
            raise ValueError(
                f"{' and '.join(requested)} needs a decoder with timestamp "
                "tokens, and this checkpoint is CTC-only"
            )
        if adaptive_timestamp and "decoder" not in self.beam_search.full_scorers:
            # The adaptive rule is applied through the decoder score, whose
            # weight is 1 - ctc_weight. At weight zero, BeamSearch drops the
            # decoder scorer and the timestamp rules with it.
            # speaker_change_symbol alone is not affected, because it replaces
            # scorefilter, which has a fixed weight.
            raise ValueError(
                "adaptive_timestamp applies the timestamp rules through the "
                "decoder score, and ctc_weight=1.0 gives the decoder no weight"
            )

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
        self._keep_special_tokens(speaker_change_symbol)

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
            # registered a second time. The mask is now part of the decoder
            # score, which is weighted by 1 - ctc_weight. A score of -inf stays
            # -inf under any positive weight, and __init__ rejects weight zero.
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
                # The filter is not a module. The ScoreFilter of the parent
                # would otherwise stay on the device without being used.
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

    def _keep_special_tokens(self, speaker_change_symbol: Optional[str]) -> None:
        """Make a hypothesis' text keep its timestamps and separator.

        For a Whisper vocabulary the parent builds a converter that drops
        special tokens, so the text would lose the timestamps these rules
        exist for. It is rebuilt here as the training preprocessor built it,
        keeping them. A token-list converter already keeps every token.

        Raises:
            ValueError: If a symbol the text is read by has another id in the
                checkpoint's token list than in the rebuilt converter.
        """
        if not isinstance(self.converter, _DroppingConverter):
            return
        separator = (
            {}
            if speaker_change_symbol is None
            else {"sot": True, "speaker_change_symbol": speaker_change_symbol}
        )
        converter = OpenAIWhisperTokenIDConverter(
            self.converter.model_type, keep_special_tokens=True, **separator
        )
        token_list = self.s2t_model.token_list
        conf = self.s2t_train_args.preprocessor_conf
        symbols = [
            conf[key]
            for key in ("first_time_symbol", "last_time_symbol", "notime_symbol")
        ] + ([] if speaker_change_symbol is None else [speaker_change_symbol])
        for symbol in symbols:
            if converter.token2id.get(symbol) != token_list.index(symbol):
                raise ValueError(
                    f"{symbol!r} is id {token_list.index(symbol)} in the "
                    f"checkpoint's token list but {converter.token2id.get(symbol)}"
                    f" in a {self.converter.model_type} converter, so the "
                    "hypothesis text would not say what the model emitted"
                )
        self.converter = converter

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
