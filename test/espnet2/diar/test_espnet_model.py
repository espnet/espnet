import pytest
import torch

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.diar.attractor.rnn_attractor import RnnAttractor
from espnet2.diar.decoder.linear_decoder import LinearDecoder
from espnet2.diar.espnet_model import ESPnetDiarizationModel
from espnet2.layers.label_aggregation import LabelAggregate

frontend = DefaultFrontend(
    n_fft=32,
    win_length=32,
    hop_length=16,
    n_mels=10,
)

encoder = TransformerEncoder(
    input_size=10,
    input_layer="linear",
    num_blocks=1,
    linear_units=32,
    output_size=16,
    attention_heads=2,
)

decoder = LinearDecoder(
    num_spk=2,
    encoder_output_size=encoder.output_size(),
)

rnn_attractor = RnnAttractor(unit=16, encoder_output_size=encoder.output_size())

label_aggregator = LabelAggregate(
    win_length=32,
    hop_length=16,
)


@pytest.mark.parametrize(
    "frontend, encoder, decoder, label_aggregator",
    [(frontend, encoder, decoder, label_aggregator)],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("attractor", [rnn_attractor, None])
def test_single_channel_model(
    label_aggregator,
    frontend,
    encoder,
    decoder,
    attractor,
    training,
):
    inputs = torch.randn(2, 300)
    ilens = torch.LongTensor([300, 200])
    spk_labels = torch.randint(high=2, size=(2, 300, 2))
    diar_model = ESPnetDiarizationModel(
        label_aggregator=label_aggregator,
        attractor=attractor,
        encoder=encoder,
        decoder=decoder,
        frontend=frontend,
        specaug=None,
        normalize=None,
    )

    if training:
        diar_model.train()
    else:
        diar_model.eval()

    kwargs = {
        "speech": inputs,
        "speech_lengths": ilens,
        "spk_labels": spk_labels,
        "spk_labels_lengths": ilens,
    }
    loss, stats, weight = diar_model(**kwargs)


def test_calc_diarization_error_counts():
    """Hand-counted example; the padded frame (index 3 of utt 1) is ignored."""
    # utt 0: 4 frames, utt 1: 3 frames + 1 padded frame; 2 speakers
    label = torch.tensor(
        [
            [[1, 0], [1, 1], [0, 0], [0, 1]],
            [[0, 0], [1, 0], [0, 1], [1, 1]],
        ],
        dtype=torch.float,
    )
    # logits: > 0 means active
    pred = torch.tensor(
        [
            [[1, -1], [1, -1], [1, -1], [-1, 1]],
            [[-1, -1], [-1, 1], [-1, 1], [1, 1]],
        ],
        dtype=torch.float,
    )
    length = torch.tensor([4, 3])
    (
        correct,
        num_frames,
        speech_scored,
        speech_miss,
        speech_falarm,
        speaker_scored,
        speaker_miss,
        speaker_falarm,
        speaker_error,
    ) = ESPnetDiarizationModel.calc_diarization_error(pred, label, length)
    assert num_frames == 7
    assert speech_scored == 5.0  # frames with any reference speaker
    assert speech_miss == 0.0
    assert speech_falarm == 1.0  # utt 0 frame 2
    assert speaker_scored == 6.0
    assert speaker_miss == 1.0  # utt 0 frame 1 has 2 speakers, 1 predicted
    assert speaker_falarm == 1.0  # utt 0 frame 2
    assert speaker_error == 1.0  # utt 1 frame 1: speaker 0 predicted as 1
    assert correct == 5.0  # (12 matching speaker slots - 2) / 2 speakers


def test_pit_loss_reorders_labels_by_best_permutation():
    """Each utterance's label comes back in its own best speaker order."""
    model = ESPnetDiarizationModel(
        label_aggregator=LabelAggregate(),
        attractor=None,
        encoder=TransformerEncoder(input_size=40, output_size=4, num_blocks=1),
        decoder=LinearDecoder(encoder_output_size=4, num_spk=2),
        frontend=DefaultFrontend(n_mels=40),
        specaug=None,
        normalize=None,
    )
    label = torch.tensor(
        [[[1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1]]], dtype=torch.float
    )
    # utt 0 predicts the labels in order, utt 1 with the speakers swapped
    pred = torch.stack([label[0], label[1][:, [1, 0]]]) * 20 - 10
    length = torch.tensor([3, 3])
    loss, min_idx, permute_list, label_perm = model.pit_loss(pred, label, length)
    assert min_idx.tolist() == [0, 1]
    assert torch.equal(label_perm[0], label[0])
    assert torch.equal(label_perm[1], label[1][:, [1, 0]])
    assert label_perm.device == pred.device


def _small_model(num_spk):
    return ESPnetDiarizationModel(
        label_aggregator=LabelAggregate(win_length=32, hop_length=16),
        attractor=None,
        encoder=TransformerEncoder(
            input_size=10,
            input_layer="linear",
            num_blocks=1,
            linear_units=32,
            output_size=16,
            attention_heads=2,
        ),
        decoder=LinearDecoder(encoder_output_size=16, num_spk=num_spk),
        frontend=DefaultFrontend(n_fft=32, win_length=32, hop_length=16, n_mels=10),
        specaug=None,
        normalize=None,
    )


def test_forward_stats_are_tensors_with_the_scored_ratios():
    """The statistics stay tensors (no copy to the CPU in forward) and equal
    the ratios of the error counts."""
    torch.manual_seed(0)
    model = _small_model(num_spk=2).eval()
    speech = torch.randn(2, 300)
    lengths = torch.tensor([300, 200])
    labels = torch.randint(0, 2, (2, 300, 2)).float()
    _, stats, _ = model(
        speech=speech,
        speech_lengths=lengths,
        spk_labels=labels,
        spk_labels_lengths=lengths,
    )
    # force_gatherable makes each statistic a 1-element tensor for gathering
    for key in ("sad_mr", "sad_fr", "mi", "fa", "cf", "acc", "der"):
        assert isinstance(stats[key], torch.Tensor), key
        assert stats[key].numel() == 1, key
    stats = {k: v.reshape(()) for k, v in stats.items() if v is not None}

    encoder_out, encoder_out_lens = model.encode(speech, lengths, None, None)
    pred = model.decoder(encoder_out, encoder_out_lens)
    agg, _ = model.label_aggregator(labels, lengths)
    _, _, _, label_perm = model.pit_loss(pred, agg, encoder_out_lens)
    c, n, ss, sm, sf, ks, km, kf, ke = model.calc_diarization_error(
        pred, label_perm, encoder_out_lens
    )
    torch.testing.assert_close(stats["sad_mr"], sm / ss)
    torch.testing.assert_close(stats["sad_fr"], sf / ss)
    torch.testing.assert_close(stats["mi"], km / ks)
    torch.testing.assert_close(stats["fa"], kf / ks)
    torch.testing.assert_close(stats["cf"], ke / ks)
    torch.testing.assert_close(stats["acc"], c / n)
    torch.testing.assert_close(stats["der"], (km + kf + ke) / ks)


def test_forward_stats_are_zero_without_reference_speech():
    """Nothing scored (all-silent labels): the ratios are 0, not NaN."""
    torch.manual_seed(0)
    model = _small_model(num_spk=1).eval()
    lengths = torch.tensor([300, 300])
    _, stats, _ = model(
        speech=torch.randn(2, 300),
        speech_lengths=lengths,
        spk_labels=torch.zeros(2, 300, 1),
        spk_labels_lengths=lengths,
    )
    for key in ("sad_mr", "sad_fr", "mi", "fa", "cf", "acc", "der"):
        assert float(stats[key]) == 0.0, key
