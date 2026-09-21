from unittest.mock import patch

import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.partially_AR_model import PartiallyARInference
from espnet2.legacy.nets.beam_search import Hypothesis


def build_model(cap):
    torch.manual_seed(0)
    decoder = TransformerDecoder(5, 4, linear_units=4, num_blocks=1)
    ctc = CTC(odim=5, encoder_output_size=4)
    model = PartiallyARInference(
        ctc=ctc,
        decoder=decoder,
        threshold_probability=0.9,
        sos=4,
        eos=4,
        mask_token=5,
        token_list=["<blank>", "<unk>", "b", "c", "<eos>"],
        weights={"decoder": 1.0},
        scorers={"decoder": decoder},
        beam_size=2,
        max_seq_len=3,
        max_mask_parallel=cap,
    ).eval()
    return model, ctc


def ctc_log_probs(ids, confidence):
    probs = torch.empty(1, len(ids), 5)
    for i, (token, probability) in enumerate(zip(ids, confidence)):
        probs[0, i].fill_((1 - probability) / 4)
        probs[0, i, token] = probability
    return probs.log()


@pytest.mark.parametrize("cap", [-1, 1, 2, 3])
@pytest.mark.parametrize("trailing_mask", [False, True])
def test_mask_contexts_and_repeated_calls(cap, trailing_mask):
    model, ctc = build_model(cap)
    # Exercise separate low-confidence spans, including a merged two-token span.
    for ids, confidence, expected in [
        ([1, 2], [0.96, 0.96], []),
        ([1, 2], [0.6, 0.96], [([4], 2)]),
        (
            [1, 3, 2, 1] + ([] if trailing_mask else [3]),
            [0.6, 0.6, 0.96, 0.6] + ([] if trailing_mask else [0.96]),
            [([4], 2), ([4, 1, 3, 2], 4 if trailing_mask else 3)],
        ),
        (
            [1, 2, 1, 3, 1, 2, 1, 3],
            [0.6, 0.96] * 4,
            [
                ([4], 2),
                ([4, 1, 2], 3),
                ([4, 1, 2, 1, 3], 2),
                ([4, 1, 2, 1, 3, 1, 2], 3),
            ],
        ),
    ]:
        with (
            patch.object(
                ctc, "log_softmax", return_value=ctc_log_probs(ids, confidence)
            ),
            patch.object(
                model.beam_search, "add_mask", wraps=model.beam_search.add_mask
            ) as add_mask,
            torch.no_grad(),
        ):
            result = model(torch.randn(len(ids), 4))
        assert [call.args for call in add_mask.call_args_list] == expected
        assert model.max_mask_parallel == cap
        assert len(result) == 1
        assert result[0].yseq[0] == model.mask_token
        assert result[0].yseq[-1] == model.mask_token


@pytest.mark.parametrize("cap", [-1, 1])
def test_hypotheses_fill_merged_masks_in_order(cap):
    model, ctc = build_model(cap)
    # Tokens after blank removal: [b, b, <unk>, c, b, <unk>], where the
    # repeated confident "b" must survive and "<unk> c" merges into one mask.
    ids = [2, 0, 2, 1, 3, 2, 1]
    confidence = [0.96, 0.96, 0.96, 0.6, 0.6, 0.96, 0.6]
    fills = {(4, 2, 2): 3, (4, 2, 2, 1, 3, 2): 1}

    def fake_beam_search(x, max_seq_len):
        return [
            Hypothesis(yseq=torch.tensor(primer + [fills[tuple(primer)], eos]))
            for primer, eos in model.beam_search.masks
        ]

    with (
        patch.object(ctc, "log_softmax", return_value=ctc_log_probs(ids, confidence)),
        patch.object(model.beam_search, "forward", side_effect=fake_beam_search),
        torch.no_grad(),
    ):
        result = model(torch.randn(len(ids), 4))
    assert result[0].yseq.tolist() == [5, 2, 2, 3, 2, 1, 5]
