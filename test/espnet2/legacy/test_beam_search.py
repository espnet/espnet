from argparse import Namespace
from pathlib import Path

import numpy
import pytest
import torch

from espnet2.legacy.nets.beam_search import BeamSearch, Hypothesis
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus
from espnet2.lm.transformer_lm import TransformerLM
from espnet2.tasks.asr import ASRTask

rnn_args = Namespace(
    encoder="rnn",
    decoder="rnn",
    input_size=8,
    encoder_conf=dict(
        # input_size=8,
        num_layers=1,
        hidden_size=2,
    ),
    specaug=None,
    normalize="utterance_mvn",
    normalize_conf={},
    decoder_conf=dict(
        hidden_size=2,
        num_layers=1,
    ),
    token_list=["a", "e", "i", "o", "u"],
    odim=5,
    mtlalpha=0.0,
    ctc_weight=0.2,
    ctc_conf={},
    init=None,
    ignore_id=-1,
    model_conf=dict(
        ctc_weight=0.2,
        ignore_id=-1,
    ),
)
transformer_args = Namespace(
    encoder="transformer",
    decoder="transformer",
    input_size=8,
    encoder_conf=dict(
        # input_size=8,
        attention_heads=2,
        linear_units=2,
        num_blocks=1,
        dropout_rate=0.0,
    ),
    specaug=None,
    normalize="utterance_mvn",
    normalize_conf={},
    decoder_conf=dict(
        attention_heads=2,
        linear_units=2,
        num_blocks=1,
        dropout_rate=0.0,
    ),
    token_list=["a", "e", "i", "o", "u"],
    odim=5,
    mtlalpha=0.0,
    ctc_weight=0.2,
    ctc_conf={},
    init=None,
    ignore_id=-1,
    model_conf=dict(
        ctc_weight=0.2,
        ignore_id=-1,
    ),
)
ldconv_args = Namespace(
    **vars(transformer_args),
)


def prepare(args, mtlalpha=0.0):
    args.mtlalpha = mtlalpha
    args.token_list = ["a", "e", "i", "o", "u"]
    args.odim = len(args.token_list)
    model = ASRTask.build_model(args)

    batchsize = 2
    x = torch.randn(batchsize, 20, 8)
    ilens = [20, 15]
    n_token = args.odim - 1
    y = (torch.rand(batchsize, 10) * n_token % (n_token - 1)).long() + 1
    olens = [10, 2]
    for i in range(batchsize):
        x[i, ilens[i] :] = -1
        y[i, olens[i] :] = -1

    data = []
    for i in range(batchsize):
        data.append(
            (
                "utt%d" % i,
                {
                    "input": [{"shape": [ilens[i], 8]}],
                    "output": [{"shape": [olens[i]]}],
                },
            )
        )
    return model, x, torch.tensor(ilens), y, data, args


@pytest.mark.parametrize(
    "args, mtlalpha, ctc_weight, lm_weight, bonus, device, dtype",
    [
        (args, ctc_train, ctc_recog, lm, bonus, device, dtype)
        for device in ("cpu", "cuda")
        for args in (transformer_args, ldconv_args, rnn_args)
        for ctc_train in (0.0, 0.5, 1.0)
        for ctc_recog in (0.0, 0.5, 1.0)
        for lm in (0.5,)
        for bonus in (0.1,)
        for dtype in ("float16", "float32", "float64")
    ],
)
def test_beam_search_equal(args, mtlalpha, ctc_weight, lm_weight, bonus, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    if device == "cpu" and dtype == "float16":
        pytest.skip("cpu float16 implementation is not available in pytorch yet")
    if mtlalpha == 0.0 and ctc_weight > 0.0:
        pytest.skip("no CTC + CTC decoding.")
    if mtlalpha == 1.0 and ctc_weight < 1.0:
        pytest.skip("pure CTC + attention decoding")
    if ctc_weight == 1.0:
        pytest.skip("pure CTC beam search is not implemented")

    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dtype = getattr(torch, dtype)
    model, x, ilens, y, data, train_args = prepare(args, mtlalpha=mtlalpha)
    model.eval()
    token_list = train_args.token_list

    lm_args = Namespace(
        lm="default",
        lm_conf=dict(
            unit=2,
            layer=1,
            embed_unit=2,
            dropout_rate=0.0,
        ),
        token_list=token_list,
    )
    lm = TransformerLM(len(token_list), **lm_args.lm_conf)
    lm.eval()

    recog_args = Namespace(
        beam_size=3,
        penalty=bonus,
        ctc_weight=ctc_weight,
        maxlenratio=0,
        lm_weight=lm_weight,
        minlenratio=0,
        nbest=3,
    )

    feat = x[0, : ilens[0]].unsqueeze(0)  # (1, T, D)
    feat_lengths = ilens[:1]  # (1,)
    model.to(device, dtype=dtype)
    model.eval()

    decoder = model.decoder

    scorers = {}
    scorers["decoder"] = decoder
    if lm_weight != 0:
        scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))

    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=recog_args.lm_weight,
        length_bonus=recog_args.penalty,
    )
    beam = BeamSearch(
        beam_size=recog_args.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        token_list=token_list,
        sos=model.sos,
        eos=model.eos,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
    beam.to(device, dtype=dtype)
    beam.eval()
    with torch.no_grad():
        print(feat_lengths)
        enc, _ = model.encode(
            torch.as_tensor(feat).to(device, dtype=dtype),
            torch.as_tensor(feat_lengths).to(device, dtype=torch.int32),
        )
        nbest_bs = beam(
            x=enc[0],
            maxlenratio=recog_args.maxlenratio,
            minlenratio=recog_args.minlenratio,
        )
    if dtype == torch.float16:
        return

    for hyp in nbest_bs:
        assert hasattr(hyp, "yseq")
        assert hasattr(hyp, "score")


def test_end_detected_converts_each_ended_hypothesis_once():
    """Agree with `end_detect` over `asdict`, converting each hypothesis once.

    A hypothesis that has ended never changes, so its summary must not be
    rebuilt at every step, also when several utterances share the cache
    within a step; one that has left every list must be forgotten.
    """
    from espnet2.legacy.nets.e2e_asr_common import end_detect

    search = BeamSearch(
        scorers={"length_bonus": LengthBonus(5)},
        weights={"length_bonus": 1.0},
        beam_size=2,
        vocab_size=5,
        sos=4,
        eos=4,
    )
    utts = [[], []]  # two utterances decoded together, as in BatchBeamSearch
    n_hyps = 0
    for i in range(1, 12):
        for b, ended in enumerate(utts):
            for extra in range(1 + (i + b) % 2):
                ended.append(
                    Hypothesis(
                        yseq=torch.arange(i + extra),
                        score=torch.tensor(-float(i) - 0.5 * extra - b),
                    )
                )
                n_hyps += 1
        for ended in utts:
            assert search.end_detected(ended, i) == end_detect(
                [h.asdict() for h in ended], i
            )
    cache = search._ended_summaries
    # every hypothesis of both utterances is cached, none evicted the other
    assert len(cache) == n_hyps
    first = utts[0][0]
    summary = cache[id(first)][1]
    assert summary == {"yseq": first.yseq.tolist(), "score": float(first.score)}

    # the next step reuses the very same summary objects: nothing is rebuilt
    calls = {"n": 0}
    orig_tolist = torch.Tensor.tolist

    def counting_tolist(self):
        calls["n"] += 1
        return orig_tolist(self)

    torch.Tensor.tolist = counting_tolist
    try:
        search.end_detected(utts[0], 12)
        search.end_detected(utts[1], 12)
    finally:
        torch.Tensor.tolist = orig_tolist
    assert calls["n"] == 0
    assert cache[id(first)][1] is summary

    # a hypothesis that is not passed for a whole step is forgotten
    search.end_detected(utts[0][:3], 13)
    search.end_detected(utts[0][:3], 14)
    assert len(search._ended_summaries) == 3


def test_end_detection_is_not_delayed_by_a_primer():
    """A primer makes every hypothesis longer; it must not make the search later.

    One good hypothesis ends at the second step and a far worse one at every
    step after it. End detection needs three such steps behind it, whatever
    the search began from. With a four-token primer it used to take three
    steps longer; longer primers delayed it further.
    """
    search = BeamSearch(
        scorers={"length_bonus": LengthBonus(5)},
        weights={"length_bonus": 1.0},
        beam_size=2,
        vocab_size=5,
        sos=4,
        eos=4,
    )

    def first_detection(primer):
        search.set_hyp_primer(primer)
        ended = []
        for i in range(1, 20):
            # a hypothesis that ends at step i has i + 1 tokens after the primer
            ended.append(
                Hypothesis(
                    yseq=torch.tensor(primer + [0] * i + [4]),
                    score=torch.tensor(0.0 if i == 1 else -100.0),
                )
            )
            if search.end_detected(ended, i):
                return i
        return None

    assert first_detection([4]) == first_detection([4, 1, 2, 3]) == 6
    assert first_detection([4] + [1] * 99) == 6


class FramePosteriors(torch.nn.Module):
    """A CTC head whose input is already the frame-level log posteriors."""

    def log_softmax(self, x):
        return x


def ctc_only_search(search_class, n_frames, scorer_name="ctc"):
    """Build a search scored by CTC alone, over speech that starts at once.

    Labels 3, 5 and 2 are spoken at frames 0, 4 and 8, the rest is blank.
    Returns the search, the posteriors to decode, the labels, and their CTC
    log likelihood as `torch.nn.functional.ctc_loss` computes it.
    """
    vocab_size, eos = 8, 7
    labels = [3, 5, 2]
    posteriors = torch.full((n_frames, vocab_size), 1e-4, dtype=torch.float64)
    posteriors[:, 0] = 1.0
    for frame, label in zip((0, 4, 8), labels):
        posteriors[frame, 0] = 1e-4
        posteriors[frame, label] = 1.0
    x = torch.log(posteriors / posteriors.sum(-1, keepdim=True))
    log_likelihood = -torch.nn.functional.ctc_loss(
        x.unsqueeze(1),
        torch.tensor([labels]),
        torch.tensor([n_frames]),
        torch.tensor([len(labels)]),
        reduction="sum",
    )
    search = search_class(
        scorers={scorer_name: CTCPrefixScorer(ctc=FramePosteriors(), eos=eos)},
        weights={scorer_name: 1.0},
        beam_size=4,
        vocab_size=vocab_size,
        sos=eos,
        eos=eos,
    )
    search.eval()
    return search, x, labels, float(log_likelihood)


@pytest.mark.parametrize("scorer_name", ["ctc", "acoustic"])
@pytest.mark.parametrize("n_frames", [12, 40])
def test_ctc_scores_the_output_not_the_primer(n_frames, scorer_name):
    """A primer longer than <sos> is not output, so CTC must not count it.

    The language and task symbols of an S2T model, or a text prompt, used to
    reach `CTCPrefixScorer` along with the output. It took each of them for a
    label already emitted and put the first real one that many frames into
    the utterance, so speech in the first frames was dropped, and a
    hypothesis that grew towards the number of frames raised IndexError.
    """
    search, x, labels, log_likelihood = ctc_only_search(
        BeamSearch, n_frames, scorer_name
    )
    for primer in (None, [search.sos, 1, 4, 6]):
        search.set_hyp_primer(primer)
        with torch.no_grad():
            best = search(x)[0]
        n_primer = 1 if primer is None else len(primer)
        assert best.yseq.tolist()[n_primer:-1] == labels
        # the non-batched scorer works in float32
        numpy.testing.assert_allclose(
            float(best.scores[scorer_name]), log_likelihood, atol=1e-4
        )


@pytest.mark.parametrize("scorer_name", ["ngram", "non_ctc"])
@pytest.mark.parametrize("primer", [[0], [0, 1], [0, 4]])
def test_partial_ngram_keeps_the_same_context_as_full_ngram(primer, scorer_name):
    pytest.importorskip("kenlm")
    from espnet2.legacy.nets.scorers.ngram import NgramFullScorer, NgramPartScorer

    tokens = ["<eos>", "I", "like", "apple", "you", "love", "coffee"]
    model = str(Path(__file__).with_name("test.arpa"))
    partial = NgramPartScorer(model, tokens)
    full = NgramFullScorer(model, tokens)
    search = BeamSearch(
        scorers={scorer_name: partial},
        weights={scorer_name: 1.0},
        beam_size=2,
        vocab_size=len(tokens),
        sos=0,
        eos=0,
        pre_beam_score_key=None,
    )
    search.set_hyp_primer(primer)
    x = torch.zeros(4, 2)
    hyp = search.init_hyp(x)[0]
    scores, _ = search.score_partial(hyp, torch.arange(len(tokens)), x)
    expected, _ = full.score(hyp.yseq, full.init_state(x), x)
    torch.testing.assert_close(scores[scorer_name], expected)
