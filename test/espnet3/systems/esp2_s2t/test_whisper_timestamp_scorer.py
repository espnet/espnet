"""Tests for WhisperTimestampFilter.

The filter returns an additive 0.0 / -inf mask over the vocabulary. A value of
0.0 means "allowed", and -inf means "forbidden".

All tests use a small vocabulary with the Whisper layout. Text comes before eos,
and a contiguous timestamp range is at the end. This layout matches the real
layout measured from OpenAIWhisperTokenIDConverter. This table shows the ids.

    ids  0..9   normal text  (SEP = 5, like "????" = 25629 in whisper vocab)
    id   10     eos
    ids  11,12  other specials
    id   13     notimestamps
    ids  14..29 timestamps   (first_time=14, last_time=29)
"""

import pytest
import torch

from espnet2.legacy.nets.beam_search import BeamSearch
from espnet2.legacy.nets.scorer_interface import BatchScorerInterface
from espnet3.systems.esp2_s2t.src.sot.whisper_timestamp_scorer import (
    BatchWhisperTimestampDecoder,
    WhisperTimestampDecoder,
    WhisperTimestampFilter,
    wrap_timestamp_decoder,
)

VOCAB = 30
EOS = 10
NOTS = 13
FIRST_T = 14
LAST_T = 29
SEP = 5
SAMPLE_BEGIN = 3  # prompt length, e.g. <|sot|><|en|><|transcribe|>

PROMPT = [100, 101, 102]  # placeholder ids. The filter uses only the length.


def build(speaker_change=None, max_initial_timestamp_index=None):
    return WhisperTimestampFilter(
        first_time=FIRST_T,
        last_time=LAST_T,
        eos=EOS,
        vocab_size=VOCAB,
        sample_begin=SAMPLE_BEGIN,
        notimestamps=NOTS,
        speaker_change=speaker_change,
        max_initial_timestamp_index=max_initial_timestamp_index,
    )


def mask_of(filt, sampled):
    """Run the filter on PROMPT + sampled and return the mask."""
    y = torch.tensor(PROMPT + list(sampled), dtype=torch.long)
    score, _ = filt.score(y, None, torch.zeros(1))
    return score


def allowed(mask):
    return set(torch.nonzero(~torch.isinf(mask)).flatten().tolist())


TIMESTAMPS = set(range(FIRST_T, LAST_T + 1))
TEXT = set(range(0, EOS))


def test_a_timestamp_range_below_eos_is_rejected():
    """The constructor raises ValueError when the timestamps come before eos.

    The mask arithmetic assumes the Whisper layout. The rule "a segment is closed,
    so text must not come next" is implemented as ``mask[:eos] = -inf``. This
    slice blocks only the text tokens when the timestamps are above eos. With the
    timestamps below eos, it would also block the timestamps, and the decode would
    be wrong without an error.
    """
    with pytest.raises(ValueError, match="eos") as excinfo:
        WhisperTimestampFilter(
            first_time=2,
            last_time=8,
            eos=20,
            vocab_size=VOCAB,
            sample_begin=SAMPLE_BEGIN,
        )
    message = str(excinfo.value)
    assert "20" in message and "2" in message


def test_the_real_whisper_layout_is_accepted():
    """The constructor accepts the ids that the recipe passes.

    The ids come from the released checkpoint's token list. eos is 50257, the
    first timestamp is 50364, the last timestamp is 51864, and the vocabulary size
    is 51865.
    """
    # The test passes if the constructor does not raise. __init__ rejects any
    # layout whose eos is inside or above the timestamp range.
    WhisperTimestampFilter(
        first_time=50364,
        last_time=51864,
        eos=50257,
        vocab_size=51865,
        sample_begin=SAMPLE_BEGIN,
    )


def test_notimestamps_is_always_forbidden():
    for sampled in ([], [FIRST_T], [FIRST_T, 3], [FIRST_T, 3, 20]):
        assert torch.isinf(mask_of(build(), sampled)[NOTS])


def test_first_position_forces_a_timestamp():
    # Nothing has been generated yet, so only timestamps are allowed.
    assert allowed(mask_of(build(), [])) == TIMESTAMPS


def test_max_initial_timestamp_bounds_the_first_timestamp():
    filt = build(max_initial_timestamp_index=2)
    assert allowed(mask_of(filt, [])) == {FIRST_T, FIRST_T + 1, FIRST_T + 2}


def test_after_two_consecutive_timestamps_text_must_follow():
    # The last two tokens are timestamps, so every timestamp is forbidden.
    m = mask_of(build(), [FIRST_T, 20])
    assert allowed(m).isdisjoint(TIMESTAMPS)
    assert 3 in allowed(m)  # text is allowed


def test_after_single_closing_timestamp_text_is_forbidden():
    # ts, text, ts means that a segment has just closed. Text is forbidden, and
    # eos and later timestamps are allowed.
    m = mask_of(build(), [FIRST_T, 3, 20])
    a = allowed(m)
    assert a.isdisjoint(TEXT)
    assert EOS in a
    assert 20 in a and 25 in a


def test_timestamps_are_monotonic():
    # The open segment started at timestamp 20, so no timestamp below 20 may close
    # it. 20 may not close it either, because a segment must have a nonzero length.
    a = allowed(mask_of(build(), [FIRST_T, 3, 20, 4]))
    assert a.isdisjoint(set(range(FIRST_T, 21)))
    assert 21 in a


def test_a_closing_timestamp_may_reopen_the_one_that_closed_a_segment():
    """After a segment closes at timestamp 20, 20 is allowed again.

    The nonzero length rule must not block the next segment, which can start where
    the previous one ended. If 20 were forbidden here too, every segment would
    start one step after the end of the previous one.
    """
    assert 20 in allowed(mask_of(build(), [FIRST_T, 3, 20]))


# ---------------------------------------------------------------------------
# SOT (speaker-change) behavior
# ---------------------------------------------------------------------------


def test_separator_is_allowed_after_a_closing_timestamp():
    sampled = [FIRST_T, 3, 20]
    assert SEP not in allowed(mask_of(build(), sampled))  # off by default
    assert SEP in allowed(mask_of(build(speaker_change=SEP), sampled))


def test_separator_forbidden_when_text_must_follow():
    # After two timestamps in a row, text must follow, so no separator is allowed.
    assert SEP not in allowed(mask_of(build(speaker_change=SEP), [FIRST_T, 20]))


def test_separator_forbidden_while_a_segment_is_open():
    """The separator is forbidden while a segment is open.

    A speaker cannot change in the middle of a segment. The timestamp that opened
    the segment would reach the segment parser without a closing timestamp. Earlier
    code handled only the two cases where the last token is a timestamp, so it
    allowed this case.
    """
    filt = build(speaker_change=SEP)
    assert SEP not in allowed(mask_of(filt, [FIRST_T, 3]))  # ts, text
    assert SEP not in allowed(mask_of(filt, [FIRST_T, 3, 4]))  # ts, text, text
    # The same case inside a new speaker block, after a speaker change.
    assert SEP not in allowed(mask_of(filt, [FIRST_T, 3, 20, SEP, 16, 4]))


def test_just_after_separator_forces_a_timestamp():
    filt = build(speaker_change=SEP)
    a = allowed(mask_of(filt, [FIRST_T, 3, 20, SEP]))
    assert a == TIMESTAMPS | {EOS}  # only a timestamp or eos
    assert SEP not in a  # no consecutive separators


def test_monotonicity_resets_after_the_speaker_change_token():
    # The first speaker's block closed at timestamp 25. A new speaker must be able
    # to start near the beginning of the window again.
    sampled = [FIRST_T, 3, 25, SEP]

    # Without speaker blocks, the global rules forbid every timestamp below 25, so
    # the next speaker could not start early. Speaker blocks fix this problem.
    assert 15 not in allowed(mask_of(build(), sampled))

    # With speaker blocks, the new block may start at any timestamp.
    assert 15 in allowed(mask_of(build(speaker_change=SEP), sampled))


def test_pairing_is_scoped_to_the_current_speaker_block():
    # A new speaker block has just started with its start timestamp (16), so text
    # must be allowed next. The global rules see the separator before 16 and treat
    # 16 as the end of a segment. So they forbid text and require another
    # timestamp, which is wrong.
    sampled = [FIRST_T, 3, 25, SEP, 16]

    a_sot = allowed(mask_of(build(speaker_change=SEP), sampled))
    assert 3 in a_sot, "text must be allowed after the timestamp opening a block"
    assert a_sot.isdisjoint(TIMESTAMPS), "a second timestamp must not follow"

    a_global = allowed(mask_of(build(), sampled))
    assert 3 not in a_global, "global pairing forbids text here"


# ---------------------------------------------------------------------------
# score and batch_score agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampled",
    [[], [FIRST_T], [FIRST_T, 20], [FIRST_T, 3, 20], [FIRST_T, 3, 20, 4]],
)
def test_score_and_batch_score_agree(sampled):
    filt = build(speaker_change=SEP)
    y = torch.tensor(PROMPT + sampled, dtype=torch.long)
    single, _ = filt.score(y, None, torch.zeros(1))
    batched, _ = filt.batch_score(y.unsqueeze(0), [None], torch.zeros(1, 1))
    assert torch.equal(single, batched[0])


# ---------------------------------------------------------------------------
# The reference rules from whisper
# ---------------------------------------------------------------------------


class _WhisperCurrentRules:
    """A copy of whisper's ApplyTimestampRules as written on openai/whisper main.

    The code is copied here because nothing in this repo pins openai-whisper.
    Release 20230308 forbids only the timestamps below the last one. Every later
    release also forbids the last one itself, except where that timestamp closed a
    segment, so that a segment cannot have zero length. The filter follows the
    later rule. A test against whatever release is installed would pass or fail by
    chance.

    ``_current_rules`` below returns the installed class when it already has the
    rule, so a modern environment still checks the real package. The order of the
    rules is important. The adaptive rule at the end reads the masked logits, so
    forbidding one more timestamp can decide whether it fires. For this reason the
    class copies all the rules. A patch applied after whisper's ``apply`` would run
    after the adaptive rule, so its result could differ.
    """

    def __init__(self, tokenizer, sample_begin, max_initial_timestamp_index):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits, tokens):
        neg = -float("inf")
        tok = self.tokenizer
        if tok.no_timestamps is not None:
            logits[:, tok.no_timestamps] = neg

        for k in range(tokens.shape[0]):
            sampled = tokens[k, self.sample_begin :]
            seq = sampled.tolist()
            last_was_ts = len(seq) >= 1 and seq[-1] >= tok.timestamp_begin
            penult_was_ts = len(seq) < 2 or seq[-2] >= tok.timestamp_begin

            if last_was_ts:
                if penult_was_ts:
                    logits[k, tok.timestamp_begin :] = neg
                else:
                    logits[k, : tok.eot] = neg

            times = sampled[sampled.ge(tok.timestamp_begin)]
            if times.numel() > 0:
                if last_was_ts and not penult_was_ts:
                    last = times[-1]
                else:
                    last = times[-1] + 1
                logits[k, tok.timestamp_begin : last] = neg

        if tokens.shape[1] == self.sample_begin:
            logits[:, : tok.timestamp_begin] = neg
            if self.max_initial_timestamp_index is not None:
                allowed_to = tok.timestamp_begin + self.max_initial_timestamp_index
                logits[:, allowed_to + 1 :] = neg

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            ts_logprob = logprobs[k, tok.timestamp_begin :].logsumexp(dim=-1)
            best_text = logprobs[k, : tok.timestamp_begin].max()
            if ts_logprob > best_text:
                logits[k, : tok.timestamp_begin] = neg


class _Tok:  # minimal stub of whisper's Tokenizer
    no_timestamps = NOTS
    timestamp_begin = FIRST_T
    eot = EOS


def _installed_forces_nonzero_segments() -> bool:
    """Return True when the installed whisper already has the nonzero segment rule."""
    from whisper.decoding import ApplyTimestampRules

    ref = ApplyTimestampRules(_Tok(), SAMPLE_BEGIN, None)
    # An open segment started at timestamp 20, and text follows it. Under the
    # current rule, 20 may not close it. Under 20230308, it may.
    logits = torch.full((1, VOCAB), -10.0)
    logits[0, 3] = 10.0  # a dominant text token, so the adaptive rule does not fire
    ref.apply(logits, torch.tensor([PROMPT + [FIRST_T, 3, 20, 4]], dtype=torch.long))
    return bool(torch.isinf(logits[0, 20]))


def _current_rules(max_initial_timestamp_index=None):
    """Return whisper's current rules, from the installed package when possible."""
    from whisper.decoding import ApplyTimestampRules

    cls = (
        ApplyTimestampRules
        if _installed_forces_nonzero_segments()
        else _WhisperCurrentRules
    )
    return cls(_Tok(), SAMPLE_BEGIN, max_initial_timestamp_index)


@pytest.mark.execution_timeout(30.0)  # importing whisper takes most of the time
def test_matches_whisper_apply_timestamp_rules():
    """Without a separator, the filter gives the same mask as whisper's rules.

    whisper's ApplyTimestampRules ends with a rule that depends on the
    distribution ("if the total probability over timestamps exceeds the best text
    token, force a timestamp"). This filter does not have that rule. So the logits
    below have one dominant text token, which keeps the rule from firing. The
    prefix that closes a segment is left out. After that prefix, whisper masks all
    text, so the adaptive rule always fires and also forbids eos.
    """
    pytest.importorskip("whisper")

    ref = _current_rules()
    filt = build()

    for sampled in ([], [FIRST_T], [FIRST_T, 20], [FIRST_T, 3, 20, 4]):
        tokens = torch.tensor([PROMPT + sampled], dtype=torch.long)
        logits = torch.full((1, VOCAB), -10.0)
        logits[0, 3] = 10.0  # dominant text token
        ref.apply(logits, tokens)
        got = filt.score(tokens[0], None, torch.zeros(1))[0]
        assert torch.equal(torch.isinf(logits[0]), torch.isinf(got)), sampled


def test_set_sample_begin_moves_the_prompt_boundary():
    filt = build()
    longer = PROMPT + [FIRST_T]  # one token past the original prompt

    # With the original boundary, this token counts as generated. So a timestamp
    # has already been emitted, and text must follow.
    y = torch.tensor(longer, dtype=torch.long)
    assert allowed(filt.score(y, None, torch.zeros(1))[0]).isdisjoint(TIMESTAMPS)

    # When the token is part of the prompt, decoding is at the first position
    # again. Only a timestamp may be emitted there.
    filt.set_sample_begin(len(longer))
    assert allowed(filt.score(y, None, torch.zeros(1))[0]) == TIMESTAMPS

    with pytest.raises(ValueError):
        filt.set_sample_begin(-1)


# ---------------------------------------------------------------------------
# Integration with ESPnet's beam search. The mask must constrain the search.
# ---------------------------------------------------------------------------


class _DummyDecoder(BatchScorerInterface):
    """A decoder scorer with a deterministic pseudo-random log-prob distribution."""

    def __init__(self, n_vocab: int):
        self.n_vocab = n_vocab

    def score(self, y, state, x):
        g = torch.Generator().manual_seed(1000 + int(y.sum().item()) + len(y))
        logits = torch.randn(self.n_vocab, generator=g)
        return torch.log_softmax(logits, dim=-1), None


def _decode(speaker_change=None, beam_size=2, maxlen=10):
    primer = [0, 1, 2]
    filt = WhisperTimestampFilter(
        first_time=FIRST_T,
        last_time=LAST_T,
        eos=EOS,
        vocab_size=VOCAB,
        sample_begin=len(primer),
        notimestamps=NOTS,
        speaker_change=speaker_change,
    )
    bs = BeamSearch(
        scorers={"decoder": _DummyDecoder(VOCAB), "timestamp": filt},
        weights={"decoder": 1.0, "timestamp": 1.0},
        beam_size=beam_size,
        vocab_size=VOCAB,
        sos=0,
        eos=EOS,
        hyp_primer=primer,
    )
    hyps = bs(x=torch.zeros(4, 2), maxlenratio=-maxlen, minlenratio=0.0)
    assert len(hyps) > 0, "beam search returned no hypothesis"
    return hyps[0].yseq.tolist()[len(primer) :]


def _strip_trailing_eos(seq):
    while seq and seq[-1] == EOS:
        seq = seq[:-1]
    return seq


def test_beam_search_output_obeys_timestamp_grammar():
    seq = _strip_trailing_eos(_decode())
    assert seq, "nothing was generated"
    assert NOTS not in seq
    assert FIRST_T <= seq[0] <= LAST_T, f"must open with a timestamp: {seq}"
    times = [t for t in seq if FIRST_T <= t <= LAST_T]
    assert times == sorted(times), f"timestamps must not decrease: {seq}"


def test_beam_search_output_obeys_sot_grammar():
    seq = _strip_trailing_eos(_decode(speaker_change=SEP))
    assert seq, "nothing was generated"
    assert NOTS not in seq
    assert FIRST_T <= seq[0] <= LAST_T, f"must open with a timestamp: {seq}"
    for i, tok in enumerate(seq):
        if tok != SEP:
            continue
        if i + 1 == len(seq):
            # The filter allows eos after a separator. Here decoding reached
            # maxlen.
            continue
        nxt = seq[i + 1]
        assert (
            FIRST_T <= nxt <= LAST_T
        ), f"separator must be followed by a timestamp: {seq}"
        assert nxt != SEP, f"no consecutive separators: {seq}"
    # timestamps must be non-decreasing inside each speaker block
    for block in " ".join(map(str, seq)).split(str(SEP)):
        times = [int(t) for t in block.split() if FIRST_T <= int(t) <= LAST_T]
        assert times == sorted(times), f"non-monotonic inside a block: {seq}"


# ---------------------------------------------------------------------------
# Adaptive rule (needs the model distribution, so it wraps the decoder)
# ---------------------------------------------------------------------------


class _FixedDecoder(BatchScorerInterface, torch.nn.Module):
    """Decoder scorer returning a fixed log-probability vector."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logp = torch.log_softmax(logits, dim=-1)

    def score(self, y, state, x):
        return self.logp.clone(), None

    def batch_score(self, ys, states, xs):
        return self.logp.unsqueeze(0).repeat(ys.shape[0], 1), None


def _wrap(logits, speaker_change=None):
    filt = build(speaker_change=speaker_change)
    return wrap_timestamp_decoder(_FixedDecoder(logits), filt)


def test_adaptive_rule_forces_a_timestamp_when_timestamp_mass_dominates():
    # The probability mass is spread over many timestamps, and every text token is
    # small. So the summed timestamp probability is larger than the best text token.
    logits = torch.full((VOCAB,), -5.0)
    logits[FIRST_T : LAST_T + 1] = 1.0
    dec = _wrap(logits)
    # The last token is text, so the base rules allow text.
    y = torch.tensor(PROMPT + [FIRST_T, 3, 20, 4], dtype=torch.long)
    out, _ = dec.score(y, None, torch.zeros(1))
    assert torch.isinf(out[:FIRST_T]).all(), "all non-timestamps must be forbidden"
    assert not torch.isinf(out[FIRST_T:]).all()


def test_adaptive_rule_stays_out_of_the_way_when_text_dominates():
    logits = torch.full((VOCAB,), -20.0)
    logits[3] = 20.0  # one overwhelming text token
    dec = _wrap(logits)
    y = torch.tensor(PROMPT + [FIRST_T, 3, 20, 4], dtype=torch.long)
    out, _ = dec.score(y, None, torch.zeros(1))
    assert not torch.isinf(out[3]), "the dominant text token must survive"


def test_adaptive_rule_uses_the_masked_distribution():
    # A huge text logit that the base rules forbid must not stop the adaptive rule
    # from firing, because whisper compares the values after masking.
    logits = torch.full((VOCAB,), -5.0)
    logits[3] = 50.0  # text, but forbidden after a closing timestamp
    logits[FIRST_T : LAST_T + 1] = 1.0
    dec = _wrap(logits)
    y = torch.tensor(PROMPT + [FIRST_T, 3, 20], dtype=torch.long)  # closing ts
    out, _ = dec.score(y, None, torch.zeros(1))
    assert torch.isinf(out[3]), "a forbidden token must not veto the rule"


def test_wrapper_still_applies_the_base_rules():
    logits = torch.zeros(VOCAB)
    dec = _wrap(logits, speaker_change=SEP)
    y = torch.tensor(PROMPT + [FIRST_T, 3, 20, SEP], dtype=torch.long)
    out, _ = dec.score(y, None, torch.zeros(1))
    assert torch.isinf(out[SEP]), "no consecutive separators"
    assert not torch.isinf(out[FIRST_T:]).all(), "a timestamp must be available"


def test_wrapper_delegates_state_handling():
    class _StatefulDecoder(_FixedDecoder):
        def init_state(self, x):
            return ("INIT", x.shape)

        def select_state(self, state, i, new_id=None):
            return ("SEL", state, i)

    stateful = WhisperTimestampDecoder(_StatefulDecoder(torch.zeros(VOCAB)), build())
    x = torch.zeros(2, 2)
    # ScorerInterface.init_state returns None for any input, so a check for None
    # would pass without delegation. The sentinel values show that the wrapper
    # passes the calls on.
    assert stateful.init_state(x) == ("INIT", (2, 2))
    assert stateful.select_state("s", 1) == ("SEL", "s", 1)

    dec = _wrap(torch.zeros(VOCAB))
    ys = torch.tensor([PROMPT + [FIRST_T]], dtype=torch.long)
    out, _ = dec.batch_score(ys, [None], torch.zeros(1, 2))
    assert out.shape == (1, VOCAB)
    single, _ = dec.score(ys[0], None, torch.zeros(2))
    assert torch.allclose(single, out[0])


@pytest.mark.execution_timeout(30.0)
def test_wrapper_matches_whisper_apply_timestamp_rules_including_adaptive():
    """The wrapper forbids the same tokens as whisper, with the adaptive rule.

    ESPnet's Whisper decoder already returns ``log_softmax(logits)``, which
    differs from ``logits`` by a scalar, and ``log_softmax`` does not change under
    a constant shift. So the masked distributions are the same, and the forbidden
    positions must match exactly.
    """
    pytest.importorskip("whisper")

    ref = _current_rules()
    torch.manual_seed(0)
    for sampled in (
        [],
        [FIRST_T],
        [FIRST_T, 20],
        [FIRST_T, 3, 20],
        [FIRST_T, 3, 20, 4],
    ):
        for _ in range(3):
            logits = torch.randn(VOCAB) * 3.0
            tokens = torch.tensor([PROMPT + sampled], dtype=torch.long)
            ref_logits = logits.clone().unsqueeze(0)
            ref.apply(ref_logits, tokens)

            dec = _wrap(logits)
            got, _ = dec.score(tokens[0], None, torch.zeros(1))
            assert torch.equal(
                torch.isinf(ref_logits[0]), torch.isinf(got)
            ), f"mismatch for {sampled}"


# In this vocabulary, more tokens follow the timestamps. Whisper's vocabulary ends
# with the timestamps, but OWSM's has this layout. <notimestamps> and the
# timestamps have low ids, and text and <sc> have higher ids. The adaptive rule
# must treat both sides of the timestamp range as "not a timestamp".
TAIL_VOCAB = 40


def build_with_tail(speaker_change=None):
    return WhisperTimestampFilter(
        first_time=FIRST_T,
        last_time=LAST_T,
        eos=EOS,
        vocab_size=TAIL_VOCAB,
        sample_begin=SAMPLE_BEGIN,
        notimestamps=NOTS,
        speaker_change=speaker_change,
    )


def test_adaptive_rule_forbids_tokens_above_the_timestamp_range():
    logits = torch.full((TAIL_VOCAB,), -5.0)
    logits[FIRST_T : LAST_T + 1] = 1.0
    dec = WhisperTimestampDecoder(_FixedDecoder(logits), build_with_tail())
    y = torch.tensor(PROMPT + [FIRST_T, 3, 20, 4], dtype=torch.long)
    out, _ = dec.score(y, None, torch.zeros(1))
    assert torch.isinf(out[:FIRST_T]).all()
    assert torch.isinf(out[LAST_T + 1 :]).all(), "the tail must be forbidden too"
    # Timestamp 20 opened the segment that is still open, so the first timestamp
    # that may close it is 21.
    assert not torch.isinf(out[21 : LAST_T + 1]).any()


def test_adaptive_rule_compares_against_tokens_above_the_timestamp_range():
    # One very large token is above the timestamps. A comparison with only the
    # tokens below them would fire the rule and remove the best token.
    logits = torch.full((TAIL_VOCAB,), -20.0)
    logits[TAIL_VOCAB - 1] = 20.0
    dec = WhisperTimestampDecoder(_FixedDecoder(logits), build_with_tail())
    y = torch.tensor(PROMPT + [FIRST_T, 3, 20, 4], dtype=torch.long)
    out, _ = dec.score(y, None, torch.zeros(1))
    assert not torch.isinf(out[TAIL_VOCAB - 1]), "the dominant token must survive"
    assert not torch.isinf(out[:FIRST_T]).all(), "the rule must not fire here"


@pytest.mark.execution_timeout(30.0)
def test_wrapper_matches_whisper_over_many_prefixes():
    """Without a separator, the wrapper matches whisper on many random prefixes.

    The tests above compare with whisper on a few chosen prefixes. This test
    checks every prefix up to length two and a seeded sample of longer prefixes
    against whisper's own class. So a regression in any single rule is found even
    when the chosen prefixes miss it. The test uses the wrapper, because whisper's
    ``apply`` ends with the adaptive rule, which the bare filter does not have.
    """
    pytest.importorskip("whisper")
    import itertools
    import random

    # whisper puts the timestamps at the end of the vocabulary, and the adaptive
    # rule depends on this. So the synthetic layout must match.
    assert LAST_T + 1 == VOCAB

    ref = _current_rules()
    dec = _wrap(torch.zeros(VOCAB))  # speaker_change=None

    rng = random.Random(0)
    torch.manual_seed(0)
    alphabet = range(VOCAB)
    prefixes = [
        list(p) for n in range(3) for p in itertools.product(alphabet, repeat=n)
    ]
    prefixes += [
        [rng.randrange(VOCAB) for _ in range(rng.randint(3, 10))] for _ in range(100)
    ]

    for sampled in prefixes:
        logits = torch.randn(VOCAB) * 5
        tokens = torch.tensor([PROMPT + sampled], dtype=torch.long)
        expected = logits.clone().unsqueeze(0)
        ref.apply(expected, tokens)
        dec.decoder.logp = torch.log_softmax(logits, dim=-1)
        got, _ = dec.score(tokens[0], None, torch.zeros(1))
        assert torch.equal(
            torch.isinf(expected[0]), torch.isinf(got)
        ), f"diverged from whisper on prefix {sampled}"


def test_a_negative_sample_begin_is_rejected():
    with pytest.raises(ValueError, match="sample_begin"):
        WhisperTimestampFilter(
            first_time=FIRST_T,
            last_time=LAST_T,
            eos=EOS,
            vocab_size=VOCAB,
            sample_begin=-1,
            notimestamps=NOTS,
        )


def test_a_separator_inside_the_timestamp_range_is_rejected():
    with pytest.raises(ValueError, match="must not be a timestamp"):
        build(speaker_change=FIRST_T + 1)


def test_a_negative_max_initial_timestamp_index_is_rejected():
    with pytest.raises(ValueError, match="max_initial_timestamp_index"):
        build(max_initial_timestamp_index=-1)


def test_without_a_notimestamps_id_that_row_is_not_blocked():
    filt = WhisperTimestampFilter(
        first_time=FIRST_T,
        last_time=LAST_T,
        eos=EOS,
        vocab_size=VOCAB,
        sample_begin=SAMPLE_BEGIN,
        notimestamps=None,
    )
    # After two timestamps, text must follow. Without a notimestamps id, the id
    # NOTS is an ordinary token. Only an explicit notimestamps id may block it.
    assert NOTS in allowed(mask_of(filt, [FIRST_T, FIRST_T]))
    assert NOTS not in allowed(mask_of(build(), [FIRST_T, FIRST_T]))


def test_the_wrapper_advertises_the_interface_its_decoder_has():
    """The wrapper has the batch interface only when its decoder has it.

    Speech2Text chooses batch beam search with isinstance checks on the scorers. A
    wrapper that always had the batch interface would send a decoder without
    batch_score into batch beam search. That decoder would fail on the first
    batch_init_state call.
    """
    from espnet2.legacy.nets.scorer_interface import ScorerInterface

    class _PlainScorer(ScorerInterface):
        def score(self, y, state, x):
            return torch.zeros(VOCAB), None

    plain = wrap_timestamp_decoder(_PlainScorer(), build())
    assert isinstance(plain, WhisperTimestampDecoder)
    assert not isinstance(plain, BatchScorerInterface)
    assert not hasattr(plain, "batch_score")

    batched = wrap_timestamp_decoder(_FixedDecoder(torch.zeros(VOCAB)), build())
    assert isinstance(batched, BatchWhisperTimestampDecoder)
    assert isinstance(batched, BatchScorerInterface)
