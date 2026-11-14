"""Tests for PITAttESPnetModel."""

import itertools

import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.linear_decoder import LinearDecoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.pit_att_espnet_model import PITAttESPnetModel
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.legacy.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)


class FixedDecoder(AbsDecoder):
    """Decoder that ignores its inputs and returns fixed logits."""

    def __init__(self, output):
        """Store the logits returned by every call."""
        super().__init__()
        self.output = output

    def forward(self, encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens):
        """Return the stored logits."""
        return self.output, None


class TinyDecoder(AbsDecoder):
    """Small order-sensitive decoder that counts its forward calls."""

    def __init__(self):
        """Build the embedding and output layers."""
        super().__init__()
        self.embed = torch.nn.Embedding(5, 4)
        self.output = torch.nn.Linear(4, 5)
        self.calls = 0

    def forward(self, encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens):
        """Return logits that depend on the input tokens."""
        self.calls += 1
        hidden = self.embed(ys_in_pad) + encoder_out.mean(dim=1, keepdim=True)
        return self.output(hidden), None


def build_model(model_class=PITAttESPnetModel, **options):
    """Build a tiny attention-only model with optional overrides."""
    args = dict(
        vocab_size=5,
        token_list=["<blank>", "<eng>", "<ara>", "<jpn>", "<sos/eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=None,
        postencoder=None,
        decoder=TinyDecoder(),
        ctc=CTC(odim=5, encoder_output_size=4),
        joint_network=None,
        ctc_weight=0.0,
        lsm_weight=0.1,
        report_cer=False,
        report_wer=False,
    )
    args.update(options)
    return model_class(**args)


@pytest.mark.parametrize("normalize_length", [False, True])
def test_pit_loss_takes_per_utterance_minimum(normalize_length):
    """The loss is the per-utterance min over all orders of its tokens."""
    model = build_model(length_normalized_loss=normalize_length)
    encoder_out = torch.randn(3, 4, 4)
    targets = [[1, 2, 3], [2, 1], [3]]
    # Without length normalization, a batch of one gives the utterance sum
    utterance_loss = LabelSmoothingLoss(size=5, padding_idx=-1, smoothing=0.1)

    def order_loss(i, tokens):
        """Return the summed loss of utterance i with the given token order."""
        ys_in_pad, ys_out_pad = add_sos_eos(
            torch.tensor([tokens]), model.sos, model.eos, -1
        )
        logits, _ = model.decoder(encoder_out[i : i + 1], None, ys_in_pad, None)
        return utterance_loss(logits, ys_out_pad)

    selected = torch.stack(
        [
            min(order_loss(i, list(p)) for p in itertools.permutations(tokens))
            for i, tokens in enumerate(targets)
        ]
    )
    if normalize_length:
        expected = selected.sum() / sum(len(t) + 1 for t in targets)
    else:
        expected = selected.mean()

    loss, _, _, _ = model._calc_att_loss(
        encoder_out,
        torch.tensor([4, 4, 4]),
        torch.tensor([[1, 2, 3], [2, 1, -1], [3, -1, -1]]),
        torch.tensor([3, 2, 1]),
    )
    torch.testing.assert_close(loss, expected)


def test_pit_loss_equals_ordered_loss_for_one_token_targets():
    """With 1-token targets the loss equals that of ESPnetASRModel."""
    model = build_model()
    base = build_model(ESPnetASRModel, decoder=model.decoder)
    args = (torch.randn(2, 3, 4), torch.tensor([3, 3]))
    targets = (torch.tensor([[1], [3]]), torch.tensor([1, 1]))
    loss, _, _, _ = model._calc_att_loss(*args, *targets)
    base_loss, _, _, _ = base._calc_att_loss(*args, *targets)
    torch.testing.assert_close(loss, base_loss)


def test_pit_loss_runs_decoder_twice_only_for_two_token_targets():
    """The swapped-order decoder pass is skipped for all 1-token batches."""
    model = build_model()
    encoder_out = torch.randn(2, 3, 4)
    model._calc_att_loss(
        encoder_out,
        torch.tensor([3, 3]),
        torch.tensor([[1], [3]]),
        torch.tensor([1, 1]),
    )
    assert model.decoder.calls == 1
    model._calc_att_loss(
        encoder_out,
        torch.tensor([3, 3]),
        torch.tensor([[1, 2], [3, -1]]),
        torch.tensor([2, 1]),
    )
    assert model.decoder.calls == 3


def test_pit_loss_accuracy_uses_the_chosen_order():
    """Token accuracy is computed with the order chosen per utterance."""
    # Utterance 0 is decoded in the swapped order, utterance 1 in the given one.
    logits = torch.full((2, 3, 5), -10.0)
    logits[0, torch.arange(3), torch.tensor([2, 1, 4])] = 10.0
    logits[1, torch.arange(3), torch.tensor([1, 2, 4])] = 10.0
    model = build_model(lsm_weight=0.0, decoder=FixedDecoder(logits))
    loss, accuracy, _, _ = model._calc_att_loss(
        torch.zeros(2, 2, 4),
        torch.tensor([2, 2]),
        torch.tensor([[1, 2], [1, 2]]),
        torch.tensor([2, 2]),
    )
    assert loss.item() == 0.0
    assert accuracy == 1.0


@pytest.mark.parametrize("max_pit_tokens", [1, 2, 3])
def test_max_pit_tokens(max_pit_tokens):
    """Targets of max_pit_tokens tokens work and longer ones raise an error."""
    model = build_model(max_pit_tokens=max_pit_tokens)
    encoder_out = torch.randn(1, 3, 4)
    for length in (max_pit_tokens, max_pit_tokens + 1):
        args = (
            encoder_out,
            torch.tensor([3]),
            torch.tensor([[1, 2, 3, 1][:length]]),
            torch.tensor([length]),
        )
        if length > max_pit_tokens:
            with pytest.raises(ValueError, match="exceeds max_pit_tokens"):
                model._calc_att_loss(*args)
        else:
            loss, _, _, _ = model._calc_att_loss(*args)
            assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"ctc_weight": 0.5}, "attention-only"),
        ({"joint_network": torch.nn.Linear(4, 5)}, "transducer"),
        ({"decoder": LinearDecoder(5, 4)}, "LinearDecoder"),
        ({"lang_token_id": 1}, "lang_token_id"),
    ],
    ids=["ctc", "transducer", "linear_decoder", "lang_token_id"],
)
def test_rejects_unsupported_options(options, message):
    """CTC, a transducer, LinearDecoder and lang_token_id are rejected."""
    with pytest.raises(ValueError, match=message):
        build_model(**options)
