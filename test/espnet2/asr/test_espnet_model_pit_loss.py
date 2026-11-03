import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.linear_decoder import LinearDecoder
from espnet2.asr.espnet_model import ESPnetASRModel
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


def build_model(**options):
    """Build a tiny attention-only ESPnetASRModel with optional overrides."""
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
    return ESPnetASRModel(**args)


@pytest.mark.parametrize("normalize_length", [False, True])
def test_pit_loss_takes_per_utterance_minimum(normalize_length):
    """The PIT loss is the per-utterance min over the two token orders."""
    model = build_model(pit_loss=True, length_normalized_loss=normalize_length)
    encoder_out = torch.randn(2, 4, 4)
    ys_pad = torch.tensor([[1, 2], [3, -1]])
    # Without length normalization, a batch of one gives the utterance sum
    utterance_loss = LabelSmoothingLoss(size=5, padding_idx=-1, smoothing=0.1)

    def per_utterance_losses(targets):
        """Return the summed loss of each utterance and the decoder targets."""
        ys_in_pad, ys_out_pad = add_sos_eos(targets, model.sos, model.eos, -1)
        logits, _ = model.decoder(encoder_out, None, ys_in_pad, None)
        losses = [
            utterance_loss(logits[i : i + 1], ys_out_pad[i : i + 1]) for i in range(2)
        ]
        return torch.stack(losses), ys_out_pad

    original, ys_out_pad = per_utterance_losses(ys_pad)
    swapped, _ = per_utterance_losses(torch.tensor([[2, 1], [3, -1]]))
    selected = torch.stack((torch.minimum(original[0], swapped[0]), original[1]))
    if normalize_length:
        expected = selected.sum() / (ys_out_pad != -1).sum()
    else:
        expected = selected.mean()

    loss, _, _, _ = model._calc_att_loss(
        encoder_out, torch.tensor([4, 4]), ys_pad, torch.tensor([2, 1])
    )
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("options", [{}, {"pit_loss": False}])
def test_pit_loss_off_uses_original_ordered_criterion(options):
    """Without pit_loss the plain ordered label-smoothing loss is used."""
    model = build_model(**options)
    encoder_out = torch.randn(2, 3, 4)
    targets = torch.tensor([[1, 2], [3, -1]])
    lengths = torch.tensor([2, 1])
    loss, _, _, _ = model._calc_att_loss(
        encoder_out, torch.tensor([3, 3]), targets, lengths
    )
    assert model.decoder.calls == 1
    inputs, outputs = add_sos_eos(targets, model.sos, model.eos, model.ignore_id)
    logits, _ = model.decoder(encoder_out, torch.tensor([3, 3]), inputs, lengths + 1)
    torch.testing.assert_close(
        loss, model.criterion_att(logits, outputs), rtol=0, atol=0
    )


def test_pit_loss_runs_decoder_twice_only_for_two_token_targets():
    """The swapped-order decoder pass is skipped for all 1-token batches."""
    model = build_model(pit_loss=True)
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
    model = build_model(pit_loss=True, lsm_weight=0.0, decoder=FixedDecoder(logits))
    loss, accuracy, _, _ = model._calc_att_loss(
        torch.zeros(2, 2, 4),
        torch.tensor([2, 2]),
        torch.tensor([[1, 2], [1, 2]]),
        torch.tensor([2, 2]),
    )
    assert loss.item() == 0.0
    assert accuracy == 1.0


def test_pit_loss_rejects_more_than_two_tokens_at_runtime():
    """Targets longer than two tokens raise a RuntimeError."""
    model = build_model(pit_loss=True)
    with pytest.raises(RuntimeError, match="only supports 1- or 2-token"):
        model._calc_att_loss(
            torch.randn(1, 3, 4),
            torch.tensor([3]),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([3]),
        )


@pytest.mark.parametrize(
    "options,message",
    [
        ({"ctc_weight": 0.5}, "attention-only"),
        ({"joint_network": torch.nn.Linear(4, 5)}, "not a transducer"),
        ({"decoder": LinearDecoder(5, 4)}, "not LinearDecoder"),
        ({"lang_token_id": 1}, "does not support lang_token_id"),
    ],
    ids=["ctc", "transducer", "linear_decoder", "lang_token_id"],
)
def test_pit_loss_rejects_unsupported_options(options, message):
    """pit_loss with CTC, a transducer, LinearDecoder or lang_token_id is rejected."""
    with pytest.raises(ValueError, match=message):
        build_model(pit_loss=True, **options)
