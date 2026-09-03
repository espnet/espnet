from test.espnet2.legacy.test_beam_search import prepare, transformer_args

import pytest
import torch

from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus


def _available(device):
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


def _decode(model, x, ilens, token_list, model_device, scoring_device):
    model = model.to(model_device).eval()
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(
            ctc=model.ctc, eos=model.eos, scoring_device=scoring_device
        ),
        "length_bonus": LengthBonus(len(token_list)),
    }
    search = BatchBeamSearch(
        beam_size=3,
        vocab_size=len(token_list),
        weights=dict(decoder=0.5, ctc=0.5, length_bonus=0.1),
        scorers=scorers,
        token_list=token_list,
        sos=model.sos,
        eos=model.eos,
        pre_beam_score_key="decoder",
    )
    search.to(model_device).eval()
    with torch.no_grad():
        feat = x[:1, : ilens[0]].to(model_device)
        enc, enc_lens = model.encode(feat, ilens[:1].to(model_device))
        return search(x=enc[0, : enc_lens[0]], maxlenratio=0.0, minlenratio=0.0)


@pytest.mark.parametrize("model_device", ["cpu", "cuda", "mps"])
def test_ctc_prefix_scorer_on_another_device_gives_the_same_result(model_device):
    """Scoring the CTC prefix on the CPU must not change what is decoded.

    The scorer moves the posteriors to its device once, keeps its states
    there, and hands the scores back on the encoder's device, so the beam
    search cannot tell the difference except through floating point.
    """
    if not _available(model_device):
        pytest.skip(f"no {model_device} device is available")
    torch.manual_seed(123)
    model, x, ilens, y, data, train_args = prepare(transformer_args, mtlalpha=0.5)
    token_list = train_args.token_list

    ref = _decode(model, x, ilens, token_list, model_device, None)
    out = _decode(model, x, ilens, token_list, model_device, "cpu")

    assert [h.yseq.tolist() for h in ref] == [h.yseq.tolist() for h in out]
    for r, o in zip(ref, out):
        torch.testing.assert_close(
            o.score.float().cpu(), r.score.float().cpu(), rtol=1e-4, atol=1e-4
        )
        # the CTC states of the finished hypotheses live on the scoring device
        assert o.states["ctc"][0].device.type == "cpu"
        assert r.states["ctc"][0].device.type == torch.device(model_device).type


def test_ctc_prefix_scorer_returns_scores_on_the_encoder_device():
    """Scores come back where the other scorers put theirs."""
    torch.manual_seed(0)
    model, x, ilens, y, data, train_args = prepare(transformer_args, mtlalpha=0.5)
    model.eval()
    scorer = CTCPrefixScorer(ctc=model.ctc, eos=model.eos, scoring_device="cpu")
    with torch.no_grad():
        enc, enc_lens = model.encode(x[:1, : ilens[0]], ilens[:1])
        enc = enc[0, : enc_lens[0]]
        scorer.batch_init_state(enc)
        assert scorer.impl.device.type == "cpu"
        ys = torch.tensor([[model.sos], [model.sos]])
        ids = torch.tensor([[1, 2], [2, 3]])
        scores, state = scorer.batch_score_partial(ys, ids, [None, None], enc)
    assert scores.device == enc.device
    assert scores.shape == (2, len(train_args.token_list))
    # the batched state can be reordered on the scoring device as well
    new_state = scorer.batch_select_state(
        state, torch.tensor([[0 * scores.size(1) + 1, 1 * scores.size(1) + 3]])
    )
    assert new_state[0].device.type == "cpu"


def test_ctc_prefix_scorer_without_a_scoring_device_is_unchanged():
    torch.manual_seed(0)
    model, x, ilens, y, data, train_args = prepare(transformer_args, mtlalpha=0.5)
    model.eval()
    scorer = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
    assert scorer.scoring_device is None
    with torch.no_grad():
        enc, enc_lens = model.encode(x[:1, : ilens[0]], ilens[:1])
        scorer.batch_init_state(enc[0, : enc_lens[0]])
    assert scorer.impl.device.type == enc.device.type
