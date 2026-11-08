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


def test_wav_augment_is_applied_in_training_only(monkeypatch):
    """The model calls wav_augment in both modes; the module augments in training.

    In training the encoder gets wav_augment's output; in eval, the input.
    """
    calls = []

    class Recorder(torch.nn.Module):
        def forward(self, speech, speech_lengths):
            calls.append((self.training, speech.shape))
            if not self.training:
                return speech, speech_lengths
            return speech + 1.0, speech_lengths

    model = ESPnetDiarizationModel(
        label_aggregator=label_aggregator,
        attractor=None,
        encoder=encoder,
        decoder=decoder,
        frontend=frontend,
        specaug=None,
        normalize=None,
        wav_augment=Recorder(),
    )
    seen = []
    encode = model.encode

    def recording_encode(speech, speech_lengths, *args):
        seen.append((speech.clone(), speech_lengths.clone()))
        return encode(speech, speech_lengths, *args)

    monkeypatch.setattr(model, "encode", recording_encode)
    speech = torch.randn(2, 300)
    kwargs = {
        "speech": speech,
        "speech_lengths": torch.LongTensor([300, 200]),
        "spk_labels": torch.randint(high=2, size=(2, 300, 2)),
        "spk_labels_lengths": torch.LongTensor([300, 200]),
    }
    model.train()
    model(**kwargs)
    assert calls == [(True, torch.Size([2, 300]))]
    torch.testing.assert_close(seen[0][0], speech + 1.0)
    model.eval()
    model(**kwargs)
    assert calls[1] == (False, torch.Size([2, 300]))
    torch.testing.assert_close(seen[1][0], speech)


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
    """The forward statistics are tensors equal to the ratios of the counts.

    They stay tensors: forward makes no copy to the CPU.
    """
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


def test_forward_does_not_copy_to_the_cpu(monkeypatch):
    """No .item(), .tolist(), .numpy() or .cpu() in forward.

    On the GPU each of them would make the CPU wait; forbidding them here
    catches a regression even where CI has no GPU.
    """
    torch.manual_seed(0)
    # features as input (no frontend): this checks the diarization model itself
    model = ESPnetDiarizationModel(
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
        decoder=LinearDecoder(encoder_output_size=16, num_spk=2),
        frontend=None,
        specaug=None,
        normalize=None,
    ).train()
    # 19 feature frames; labels of 300 samples aggregate to the same 19 frames
    kwargs = dict(
        speech=torch.randn(2, 19, 10),
        speech_lengths=torch.tensor([19, 12]),
        spk_labels=torch.randint(0, 2, (2, 300, 2)).float(),
        spk_labels_lengths=torch.tensor([300, 200]),
    )

    def forbidden(self, *args, **kw):
        raise AssertionError("forward copied a tensor to the CPU")

    for name in ("item", "tolist", "numpy", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    loss, stats, weight = model(**kwargs)
    monkeypatch.undo()
    assert torch.isfinite(loss)


_accelerators = [
    d
    for d, ok in [
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    ]
    if ok
]


@pytest.mark.skipif(not _accelerators, reason="needs a CUDA or MPS device")
@pytest.mark.parametrize("device", _accelerators)
def test_pit_loss_and_counts_stay_on_the_device(device):
    """On an accelerator, the reordered labels and the counts stay there."""
    torch.manual_seed(0)
    model = _small_model(num_spk=2).to(device)
    pred = torch.randn(3, 20, 2, device=device)
    label = torch.randint(0, 2, (3, 20, 2), device=device).float()
    length = torch.tensor([20, 15, 7], device=device)
    loss, _, _, label_perm = model.pit_loss(pred, label, length)
    assert loss.device.type == device and label_perm.device.type == device
    counts = model.calc_diarization_error(pred, label_perm, length)
    assert all(c.device.type == device for c in counts)

    cpu = [t.cpu() for t in (pred, label, length)]
    _, _, _, label_perm_cpu = model.to("cpu").pit_loss(*cpu)
    torch.testing.assert_close(label_perm.cpu(), label_perm_cpu)


# --- Equivalence with the implementation on master before #6987 -------------
# Copies of the NumPy / per-utterance code that this PR replaced, used as the
# reference: the new code must give the same loss, permutation, reordered
# labels, mask, error counts and statistics.


def _old_create_length_mask(length, max_len, num_output):
    mask = torch.zeros(len(length), max_len, num_output)
    for i in range(len(length)):
        mask[i, : length[i], :] = 1
    return mask


def _old_pit_loss(pred, label, lengths):
    from itertools import permutations

    import numpy as np

    num_output = label.size(2)
    permute_list = [np.array(p) for p in permutations(range(num_output))]
    loss_list = []
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    mask = _old_create_length_mask(lengths, label.size(1), label.size(2))
    for p in permute_list:
        loss_perm = bce(pred, label[:, :, p]) * mask
        loss_list.append(torch.sum(torch.mean(loss_perm, dim=2), dim=1)[:, None])
    loss = torch.cat(loss_list, dim=1)
    min_loss, min_idx = torch.min(loss, dim=1)
    loss = torch.sum(min_loss) / torch.sum(lengths.float())
    label_list = [
        label[i, :, permute_list[min_idx[i]]].data.cpu().numpy()
        for i in range(len(min_idx))
    ]
    return loss, min_idx, torch.from_numpy(np.array(label_list)).float()


def _old_calc_diarization_error(pred, label, length):
    import numpy as np

    batch_size, max_len, num_output = label.size()
    mask = np.zeros((batch_size, max_len, num_output))
    for i in range(batch_size):
        mask[i, : length[i], :] = 1
    label_np = label.data.cpu().numpy().astype(int) * mask
    pred_np = (pred.data.cpu().numpy() > 0).astype(int) * mask
    n_ref = np.sum(label_np, axis=2)
    n_sys = np.sum(pred_np, axis=2)
    n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
    return (
        float(1.0 * np.sum((label_np == pred_np) * mask) / num_output),
        np.sum(length.data.cpu().numpy()),
        float(np.sum(n_ref > 0)),
        float(np.sum(np.logical_and(n_ref > 0, n_sys == 0))),
        float(np.sum(np.logical_and(n_ref == 0, n_sys > 0))),
        float(np.sum(n_ref)),
        float(np.sum(np.maximum(n_ref - n_sys, 0))),
        float(np.sum(np.maximum(n_sys - n_ref, 0))),
        float(np.sum(np.minimum(n_ref, n_sys) - n_map)),
    )


def _old_stats(counts):
    c, n, ss, sm, sf, ks, km, kf, ke = counts
    if ss > 0 and n > 0:
        return dict(
            sad_mr=sm / ss,
            sad_fr=sf / ss,
            mi=km / ks,
            fa=kf / ks,
            cf=ke / ks,
            acc=c / n,
            der=(km + kf + ke) / ks,
        )
    return dict(sad_mr=0, sad_fr=0, mi=0, fa=0, cf=0, acc=0, der=0)


def _random_case(g, num_spk):
    b = int(torch.randint(1, 6, (1,), generator=g))
    t = int(torch.randint(1, 41, (1,), generator=g))
    lengths = torch.randint(1, t + 1, (b,), generator=g)
    lengths[0] = t  # the longest utterance fills the batch, as after padding
    pred = torch.randn(b, t, num_spk, generator=g)
    label = (torch.rand(b, t, num_spk, generator=g) < 0.4).float()
    if b > 1:
        label[-1] = 0  # an utterance without speech
    return pred, label, lengths


@pytest.mark.parametrize("num_spk", [1, 2, 3])
def test_loss_counts_and_mask_equal_the_previous_implementation(num_spk):
    """pit_loss, create_length_mask and calc_diarization_error equal master's."""
    model = _small_model(num_spk=num_spk)
    g = torch.Generator().manual_seed(num_spk)
    for _ in range(200):
        pred, label, lengths = _random_case(g, num_spk)
        t = label.size(1)

        torch.testing.assert_close(
            model.create_length_mask(lengths, t, num_spk),
            _old_create_length_mask(lengths, t, num_spk),
        )

        loss, min_idx, _, label_perm = model.pit_loss(pred, label, lengths)
        old_loss, old_min_idx, old_label_perm = _old_pit_loss(pred, label, lengths)
        torch.testing.assert_close(loss, old_loss)
        assert torch.equal(min_idx, old_min_idx)
        assert torch.equal(label_perm, old_label_perm)

        counts = model.calc_diarization_error(pred, label_perm, lengths)
        old_counts = _old_calc_diarization_error(pred, old_label_perm, lengths)
        # the counts are integers and equal; `correct` is divided by num_output in
        # float32 on the device instead of float64 on the CPU
        assert [float(c) for c in counts[1:]] == [float(c) for c in old_counts[1:]]
        assert float(counts[0]) == pytest.approx(old_counts[0], rel=1e-6)


@pytest.mark.parametrize("num_spk", [1, 2])
def test_forward_stats_equal_the_previous_implementation(num_spk):
    """forward's loss and statistics equal master's for random batches."""
    torch.manual_seed(0)
    model = _small_model(num_spk=num_spk).eval()
    g = torch.Generator().manual_seed(10 + num_spk)
    for _ in range(20):
        b = int(torch.randint(1, 4, (1,), generator=g))
        lengths = torch.randint(64, 301, (b,), generator=g)
        lengths[0] = 300
        speech = torch.randn(b, 300, generator=g)
        labels = (torch.rand(b, 300, num_spk, generator=g) < 0.4).float()
        loss, stats, _ = model(
            speech=speech,
            speech_lengths=lengths,
            spk_labels=labels,
            spk_labels_lengths=lengths,
        )

        enc, enc_lens = model.encode(speech, lengths, None, None)
        pred = model.decoder(enc, enc_lens)
        agg, _ = model.label_aggregator(labels, lengths)
        old_loss, _, old_label_perm = _old_pit_loss(pred, agg, enc_lens)
        expected = _old_stats(
            _old_calc_diarization_error(pred, old_label_perm, enc_lens)
        )
        torch.testing.assert_close(loss.reshape(()), old_loss)
        for key, value in expected.items():
            assert float(stats[key]) == pytest.approx(value, abs=1e-6), key


def test_collect_feats_removes_the_augmentation_context(tmp_path):
    """The feature statistics are those of the windows without their context."""
    import numpy as np
    import soundfile

    from espnet2.layers.batch_wav_augment import BatchWavAugment

    soundfile.write(tmp_path / "n.wav", np.zeros(4000, dtype=np.int16), 16000)
    (tmp_path / "noise.scp").write_text(f"n {tmp_path / 'n.wav'}\n")
    aug = BatchWavAugment(noise_scp=str(tmp_path / "noise.scp"), context=100)
    model = ESPnetDiarizationModel(
        label_aggregator=label_aggregator,
        attractor=None,
        encoder=encoder,
        decoder=decoder,
        frontend=frontend,
        specaug=None,
        normalize=None,
        wav_augment=aug,
    )
    speech = torch.randn(2, 400)
    lengths = torch.LongTensor([400, 300])
    out = model.collect_feats(speech, lengths)
    feats, feats_lengths = model._extract_feats(speech[:, 100:], lengths - 100)
    torch.testing.assert_close(out["feats"], feats)
    assert out["feats_lengths"].tolist() == feats_lengths.tolist()


def test_chunks_with_left_context_train_and_validate(tmp_path):
    """Chunk iterator with left context -> model with wav_augment context.

    The speech chunks carry the context and the label chunks do not; after the
    model removes the context, the encoder frames and the label frames agree,
    in training (with reverberation) and in validation.
    """
    import numpy as np
    import soundfile

    from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
    from espnet2.layers.batch_wav_augment import BatchWavAugment
    from espnet2.train.collate_fn import CommonCollateFn

    rir = np.zeros(200, dtype=np.float32)
    rir[0], rir[50] = 1.0, 0.5
    soundfile.write(tmp_path / "rir.wav", rir, 16000, subtype="FLOAT")
    (tmp_path / "rir.scp").write_text(f"r {tmp_path / 'rir.wav'}\n")
    context, width = 400, 2000

    class Recordings:
        def __getitem__(self, item):
            n = 7000
            labels = np.zeros((n, 2), dtype=np.float32)
            labels[1000:5000, 0] = 1
            labels[3000:6000, 1] = 1
            speech = np.random.default_rng(0).normal(size=n).astype(np.float32)
            return item, {"speech": speech * 0.05, "spk_labels": labels}

    iterator = ChunkIterFactory(
        dataset=Recordings(),
        batches=[["a"], ["b"]],
        batch_size=2,
        chunk_length=width,
        chunk_shift_ratio=1.0,
        collate_fn=CommonCollateFn(),
        left_context=context,
    )
    model = ESPnetDiarizationModel(
        label_aggregator=label_aggregator,
        attractor=None,
        encoder=encoder,
        decoder=decoder,
        frontend=frontend,
        specaug=None,
        normalize=None,
        wav_augment=BatchWavAugment(rir_scp=str(tmp_path / "rir.scp"), context=context),
    )
    n = 0
    for _, batch in iterator.build_iter(0):
        assert batch["speech"].shape[1] == context + width
        assert batch["spk_labels"].shape[1] == width
        for train in (True, False):
            model.train(train)
            loss, stats, weight = model(**batch)
            assert torch.isfinite(loss)
        n += 1
    assert n > 0


# --- Look-ahead (dual-mode) TCN VAD ------------------------------------------


def _vad_model(tmp_path, look_ahead=0, look_aheads=None):
    """A VAD: STFT without centring, global MVN, TCN encoder, one output."""
    import numpy as np

    from espnet2.asr.encoder.tcn_encoder import TCNEncoder
    from espnet2.layers.global_mvn import GlobalMVN

    stats = tmp_path / "feats_stats.npz"
    np.savez(
        stats, count=np.array(100), sum=np.zeros(10), sum_square=np.full(10, 100.0)
    )
    torch.manual_seed(0)
    return ESPnetDiarizationModel(
        label_aggregator=LabelAggregate(win_length=64, hop_length=16, center=False),
        attractor=None,
        encoder=TCNEncoder(
            10,
            output_size=8,
            bottleneck_size=4,
            hidden_size=8,
            num_blocks=3,
            num_repeats=2,
            future_taps=2,
            look_ahead=look_ahead,
        ),
        decoder=LinearDecoder(encoder_output_size=8, num_spk=1),
        frontend=DefaultFrontend(
            n_fft=64, win_length=64, hop_length=16, n_mels=10, center=False
        ),
        specaug=None,
        normalize=GlobalMVN(str(stats)),
        look_aheads=look_aheads,
    ).eval()


def _vad_probs(model, speech):
    with torch.no_grad():
        lens = torch.tensor([speech.size(1)])
        enc, enc_lens = model.encode(speech, lens, None, None)
        return model.decoder(enc, enc_lens)[0, :, 0]


@pytest.mark.parametrize("look_ahead", [0, 6, 28])
def test_vad_reads_no_sample_beyond_its_look_ahead(tmp_path, look_ahead):
    """The whole VAD (frontend to output) never reads ahead of its look-ahead.

    Frame i covers samples [16 i, 16 i + 64); with look-ahead L it may read
    samples up to 16 (i + L) + 63. With L = 0 this is the online VAD. The first
    frame that may read a sample reads it at the edge of its STFT window, with a
    tiny weight, so the change is looked for in the first four such frames.
    """
    from espnet2.torch_utils.check_look_ahead import (
        check_look_ahead,
        frame_last_sample,
    )

    model = _vad_model(tmp_path, look_ahead=look_ahead)
    torch.manual_seed(1)
    check_look_ahead(
        lambda speech: _vad_probs(model, speech)[None],  # (1, frames)
        torch.randn(1, 3000) * 0.1,  # (1, samples)
        frame_last_sample(hop_length=16, win_length=64, look_ahead=look_ahead),
        range(200, 3000, 97),
        tight_window=4,
    )


def test_vad_trains_on_each_look_ahead(tmp_path):
    """look_aheads=[0, 28]: the loss is the mean of the two modes' losses."""
    model = _vad_model(tmp_path, look_aheads=[0, 28])
    speech = torch.randn(2, 3000) * 0.1
    lens = torch.tensor([3000, 2500])
    labels = (torch.rand(2, 3000, 1) > 0.5).float()
    loss, stats, _ = model(
        speech=speech,
        speech_lengths=lens,
        spk_labels=labels,
        spk_labels_lengths=lens,
    )
    assert model.encoder.look_ahead == 0  # restored
    one = {}
    for frames in (0, 28):
        model.encoder.set_look_ahead(frames)
        one[frames] = model._forward_one(speech, lens, labels, lens)[0]
    model.encoder.set_look_ahead(0)
    torch.testing.assert_close(loss.reshape(()), (one[0] + one[28]) / 2)
    torch.testing.assert_close(stats["loss_la0"].reshape(()), one[0])
    torch.testing.assert_close(stats["loss_la28"].reshape(()), one[28])
    assert {"acc", "acc_la0", "acc_la28", "sad_mr_la0"} <= set(stats)


def test_look_aheads_need_a_look_ahead_encoder():
    with pytest.raises(ValueError):
        ESPnetDiarizationModel(
            label_aggregator=label_aggregator,
            attractor=None,
            encoder=encoder,
            decoder=decoder,
            frontend=frontend,
            specaug=None,
            normalize=None,
            look_aheads=[0, 4],
        )


def test_look_aheads_must_be_reachable(tmp_path):
    with pytest.raises(ValueError):
        _vad_model(tmp_path, look_aheads=[0, 29])


def test_set_look_ahead(tmp_path):
    model = _vad_model(tmp_path)
    model.set_look_ahead(14)
    assert model.encoder.look_ahead == 14
    plain = ESPnetDiarizationModel(
        label_aggregator=label_aggregator,
        attractor=None,
        encoder=encoder,
        decoder=decoder,
        frontend=frontend,
        specaug=None,
        normalize=None,
    )
    with pytest.raises(ValueError):
        plain.set_look_ahead(0)


def test_look_aheads_backpropagate_the_mean_of_the_modes(tmp_path):
    model = _vad_model(tmp_path, look_aheads=[0, 28]).train()
    torch.manual_seed(3)
    lens = torch.tensor([3000, 2500])
    batch = dict(
        speech=torch.randn(2, 3000) * 0.1,
        speech_lengths=lens,
        spk_labels=(torch.rand(2, 3000, 1) > 0.5).float(),
        spk_labels_lengths=lens,
    )
    args = [batch[k] for k in ("speech", "speech_lengths", "spk_labels")]
    args.append(batch["spk_labels_lengths"])

    def grads(loss):
        model.zero_grad()
        loss.sum().backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    torch.manual_seed(0)
    dual = grads(model(**batch)[0])
    each = []
    for frames in (0, 28):
        model.encoder.set_look_ahead(frames)
        each.append(model._forward_one(*args)[0])
    model.encoder.set_look_ahead(0)
    mean = grads((each[0] + each[1]) / 2)
    assert len(dual) == len(mean)
    for a, b in zip(dual, mean):
        torch.testing.assert_close(a, b)


def test_failed_look_aheads_check_restores_the_encoder(tmp_path):
    from espnet2.asr.encoder.tcn_encoder import TCNEncoder

    encoder = TCNEncoder(10, num_blocks=3, num_repeats=2, future_taps=2, look_ahead=6)
    with pytest.raises(ValueError):
        ESPnetDiarizationModel(
            label_aggregator=label_aggregator,
            attractor=None,
            encoder=encoder,
            decoder=LinearDecoder(encoder_output_size=128, num_spk=1),
            frontend=None,
            specaug=None,
            normalize=None,
            look_aheads=[0, 29],
        )
    assert encoder.look_ahead == 6


def test_look_aheads_without_frontend_see_the_same_features(tmp_path):
    """Features in, GlobalMVN in place (no gradient): each mode gets them as given."""
    import numpy as np

    from espnet2.asr.encoder.tcn_encoder import TCNEncoder
    from espnet2.layers.global_mvn import GlobalMVN

    stats = tmp_path / "feats_stats.npz"
    # different statistics per dimension: the encoder's per-frame layer norm
    # would hide a second normalization with the same shift for all dimensions
    np.savez(
        stats,
        count=np.array(10),
        sum=np.arange(6) * 4.0,
        sum_square=np.arange(6) ** 2 * 3.0 + 20.0,
    )
    torch.manual_seed(0)
    model = ESPnetDiarizationModel(
        label_aggregator=LabelAggregate(win_length=1, hop_length=1, center=False),
        attractor=None,
        encoder=TCNEncoder(
            6, output_size=8, num_blocks=2, num_repeats=1, future_taps=2
        ),
        decoder=LinearDecoder(encoder_output_size=8, num_spk=1),
        frontend=None,
        specaug=None,
        normalize=GlobalMVN(str(stats)),
        look_aheads=[0, 6],
    ).eval()
    feats = torch.randn(2, 30, 6) + 3.0
    lens = torch.tensor([30, 24])
    labels = (torch.rand(2, 30, 1) > 0.5).float()
    given = feats.clone()
    with torch.no_grad():
        loss = model(feats, lens, labels, lens)[0]
        each = []
        for frames in (0, 6):
            model.encoder.set_look_ahead(frames)
            each.append(model._forward_one(given.clone(), lens, labels, lens)[0])
    torch.testing.assert_close(loss.reshape(()), (each[0] + each[1]) / 2)


def test_wav_augment_runs_once_for_all_look_aheads(tmp_path):
    """With look_aheads, every look-ahead sees the same augmented batch."""
    calls = []

    class Augment(torch.nn.Module):
        def forward(self, speech, speech_lengths):
            calls.append(self.training)
            return speech + 0.01 * torch.randn_like(speech), speech_lengths

    model = _vad_model(tmp_path, look_aheads=[0, 28]).train()
    model.wav_augment = Augment()
    lens = torch.tensor([3000, 2500])
    batch = dict(
        speech=torch.randn(2, 3000) * 0.1,
        speech_lengths=lens,
        spk_labels=(torch.rand(2, 3000, 1) > 0.5).float(),
        spk_labels_lengths=lens,
    )
    model(**batch)
    assert calls == [True]
