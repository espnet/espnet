"""Tests for KmeansModel, DiscreteAudioIO, and ContinuousAudioIO."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from espnet2.speechlm.model.speechlm.multimodal_io.audio import (
    ContinuousAudioIO,
    DiscreteAudioIO,
    KmeansModel,
)


# ---------------------------------------------------------------------------
# KmeansModel tests
# ---------------------------------------------------------------------------
class TestKmeansModel:
    def _make_km(self, centers):
        """Build KmeansModel without joblib, using manual buffer registration."""
        C_np = centers.T  # [D, K]
        Cnorm_np = (C_np**2).sum(0, keepdims=True)
        model = object.__new__(KmeansModel)
        torch.nn.Module.__init__(model)
        model.register_buffer("C", torch.from_numpy(C_np).float())
        model.register_buffer("Cnorm", torch.from_numpy(Cnorm_np).float())
        return model

    def test_call_returns_correct_indices(self):
        centers = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        km = self._make_km(centers)
        x = torch.tensor([[0.1, 0.1], [9.9, 9.9], [19.5, 19.5]])
        indices = km(x)
        assert indices.tolist() == [[0], [1], [2]]

    def test_call_rejects_non_tensor(self):
        centers = np.array([[0.0], [1.0]])
        km = self._make_km(centers)
        with pytest.raises(TypeError):
            km([0.5])

    def test_output_shape(self):
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        km = self._make_km(centers)
        x = torch.randn(7, 3)
        out = km(x)
        assert out.shape == (7, 1)

    def test_output_dtype(self):
        centers = np.array([[0.0], [1.0]])
        km = self._make_km(centers)
        out = km(torch.tensor([[0.4]]))
        assert out.dtype == torch.int64


# ---------------------------------------------------------------------------
# DiscreteAudioIO fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def ssl_only_io():
    """DiscreteAudioIO with SSL only (no codec)."""
    with (
        patch.object(DiscreteAudioIO, "_init_codec"),
        patch.object(DiscreteAudioIO, "_init_ssl"),
        patch.object(DiscreteAudioIO, "_init_sanity_check"),
    ):
        io = DiscreteAudioIO(ssl_choice="ESPnet", ssl_hf_model_tag="espnet/xeus")

    io.codec_model = None
    io.codec_n_streams = 0
    io.codec_vocab_size = []
    io.codec_sample_rate = None
    io.codec_frame_shift = None
    io.codec_frame_per_second = None

    io.ssl_model = None
    io.km_model = None
    io.ssl_n_streams = 1
    io.ssl_vocab_size = [500]
    io.ssl_sample_rate = 16000
    io.ssl_frame_shift = 320
    io.ssl_frame_per_second = 50

    io._init_sanity_check()
    return io


@pytest.fixture
def codec_only_io():
    """DiscreteAudioIO with codec only (no SSL)."""
    with (
        patch.object(DiscreteAudioIO, "_init_codec"),
        patch.object(DiscreteAudioIO, "_init_ssl"),
        patch.object(DiscreteAudioIO, "_init_sanity_check"),
    ):
        io = DiscreteAudioIO(codec_choice="ESPnet", codec_hf_model_tag="mock/codec")

    io.ssl_model = None
    io.km_model = None
    io.ssl_n_streams = 0
    io.ssl_vocab_size = []
    io.ssl_sample_rate = None
    io.ssl_frame_shift = None
    io.ssl_frame_per_second = None

    io.codec_model = None
    io.codec_n_streams = 4
    io.codec_vocab_size = [1024, 1024, 1024, 1024]
    io.codec_sample_rate = 16000
    io.codec_frame_shift = 320
    io.codec_frame_per_second = 50

    io._init_sanity_check()
    return io


@pytest.fixture
def ssl_codec_io():
    """DiscreteAudioIO with both SSL and codec."""
    with (
        patch.object(DiscreteAudioIO, "_init_codec"),
        patch.object(DiscreteAudioIO, "_init_ssl"),
        patch.object(DiscreteAudioIO, "_init_sanity_check"),
    ):
        io = DiscreteAudioIO(
            ssl_choice="ESPnet",
            ssl_hf_model_tag="espnet/xeus",
            codec_choice="ESPnet",
            codec_hf_model_tag="mock/codec",
            codec_max_token_per_frame=2,
        )

    io.ssl_model = None
    io.km_model = None
    io.ssl_n_streams = 1
    io.ssl_vocab_size = [500]
    io.ssl_sample_rate = 16000
    io.ssl_frame_shift = 320
    io.ssl_frame_per_second = 50

    io.codec_model = None
    io.codec_n_streams = 2
    io.codec_vocab_size = [1024, 1024]
    io.codec_sample_rate = 16000
    io.codec_frame_shift = 320
    io.codec_frame_per_second = 50

    io._init_sanity_check()
    return io


# ---------------------------------------------------------------------------
# DiscreteAudioIO — init validation
# ---------------------------------------------------------------------------
class TestDiscreteAudioIOInit:
    def test_no_tokenizer_raises(self):
        with pytest.raises(ValueError, match="At least one tokenizer"):
            DiscreteAudioIO()

    def test_unsupported_codec_raises(self):
        with patch.object(DiscreteAudioIO, "_init_ssl"):
            with pytest.raises(NotImplementedError, match="Cannot support codec"):
                DiscreteAudioIO(
                    codec_choice="UnsupportedCodec",
                    ssl_choice=None,
                )

    def test_unsupported_ssl_raises(self):
        with patch.object(DiscreteAudioIO, "_init_codec"):
            with pytest.raises(NotImplementedError, match="Cannot support SSL"):
                DiscreteAudioIO(
                    ssl_choice="UnsupportedSSL",
                    codec_choice=None,
                )


# ---------------------------------------------------------------------------
# DiscreteAudioIO — num_stream
# ---------------------------------------------------------------------------
class TestDiscreteAudioIONumStream:
    def test_ssl_only(self, ssl_only_io):
        assert ssl_only_io.num_stream() == 1

    def test_codec_only(self, codec_only_io):
        assert codec_only_io.num_stream() == 4

    def test_ssl_codec(self, ssl_codec_io):
        assert ssl_codec_io.num_stream() == 3  # 1 SSL + 2 codec


# ---------------------------------------------------------------------------
# DiscreteAudioIO — vocabulary
# ---------------------------------------------------------------------------
class TestDiscreteAudioIOVocabulary:
    def test_ssl_only_vocab(self, ssl_only_io):
        vocab = ssl_only_io.get_vocabulary()
        # 1 pad + 500 tokens = 501
        assert len(vocab) == 501
        assert vocab[0] == "<ssl_layer0_pad>"
        assert vocab[1] == "<ssl_layer0_0>"
        assert vocab[500] == "<ssl_layer0_499>"

    def test_codec_only_vocab(self, codec_only_io):
        vocab = codec_only_io.get_vocabulary()
        # 4 streams * (1 pad + 1024 tokens) = 4100
        assert len(vocab) == 4100
        assert vocab[0] == "<codec_layer0_pad>"
        assert vocab[1025] == "<codec_layer1_pad>"

    def test_ssl_codec_vocab_ordering(self, ssl_codec_io):
        vocab = ssl_codec_io.get_vocabulary()
        # SSL: 501, Codec: 2 * 1025 = 2050, total = 2551
        assert len(vocab) == 2551
        assert vocab[0] == "<ssl_layer0_pad>"
        assert vocab[501] == "<codec_layer0_pad>"


# ---------------------------------------------------------------------------
# DiscreteAudioIO — stream intervals
# ---------------------------------------------------------------------------
class TestDiscreteAudioIOStreamInterval:
    def test_ssl_only(self, ssl_only_io):
        intervals = ssl_only_io.get_stream_interval()
        assert intervals == [(0, 501)]

    def test_codec_only(self, codec_only_io):
        intervals = codec_only_io.get_stream_interval()
        assert len(intervals) == 4
        assert intervals[0] == (0, 1025)
        assert intervals[1] == (1025, 2050)

    def test_ssl_codec(self, ssl_codec_io):
        intervals = ssl_codec_io.get_stream_interval()
        assert len(intervals) == 3
        assert intervals[0] == (0, 501)  # SSL
        assert intervals[1] == (501, 1526)  # codec stream 0
        assert intervals[2] == (1526, 2551)  # codec stream 1


# ---------------------------------------------------------------------------
# DiscreteAudioIO — stream weights
# ---------------------------------------------------------------------------
class TestDiscreteAudioIOStreamWeight:
    def test_default_all_ones(self, ssl_only_io):
        assert ssl_only_io.get_stream_weight() == [1.0]

    def test_default_multi_stream(self, ssl_codec_io):
        assert ssl_codec_io.get_stream_weight() == [1.0, 1.0, 1.0]

    def test_custom_weights(self):
        with (
            patch.object(DiscreteAudioIO, "_init_codec"),
            patch.object(DiscreteAudioIO, "_init_ssl"),
            patch.object(DiscreteAudioIO, "_init_sanity_check"),
        ):
            io = DiscreteAudioIO(
                ssl_choice="ESPnet",
                ssl_hf_model_tag="espnet/xeus",
                codec_choice="ESPnet",
                codec_hf_model_tag="m",
                stream_weights=[0.5, 1.0, 0.8],
            )
        io.ssl_model = None
        io.km_model = None
        io.ssl_n_streams = 1
        io.ssl_vocab_size = [500]
        io.ssl_sample_rate = 16000
        io.ssl_frame_shift = 320
        io.ssl_frame_per_second = 50
        io.codec_model = None
        io.codec_n_streams = 2
        io.codec_vocab_size = [1024, 1024]
        io.codec_sample_rate = 16000
        io.codec_frame_shift = 320
        io.codec_frame_per_second = 50
        io._init_sanity_check()
        assert io.get_stream_weight() == [0.5, 1.0, 0.8]

    def test_wrong_weight_count_raises(self):
        with (
            patch.object(DiscreteAudioIO, "_init_codec"),
            patch.object(DiscreteAudioIO, "_init_ssl"),
            patch.object(DiscreteAudioIO, "_init_sanity_check"),
        ):
            io = DiscreteAudioIO(
                ssl_choice="ESPnet",
                ssl_hf_model_tag="espnet/xeus",
                stream_weights=[1.0, 2.0],  # wrong count for 1 stream
            )
        io.codec_model = None
        io.codec_n_streams = 0
        io.codec_vocab_size = []
        io.codec_sample_rate = None
        io.codec_frame_shift = None
        io.codec_frame_per_second = None
        io.ssl_model = None
        io.km_model = None
        io.ssl_n_streams = 1
        io.ssl_vocab_size = [500]
        io.ssl_sample_rate = 16000
        io.ssl_frame_shift = 320
        io.ssl_frame_per_second = 50
        with pytest.raises(ValueError, match="Number of weights"):
            io._init_sanity_check()

    def test_negative_weight_raises(self):
        with (
            patch.object(DiscreteAudioIO, "_init_codec"),
            patch.object(DiscreteAudioIO, "_init_ssl"),
            patch.object(DiscreteAudioIO, "_init_sanity_check"),
        ):
            io = DiscreteAudioIO(
                ssl_choice="ESPnet",
                ssl_hf_model_tag="espnet/xeus",
                stream_weights=[-1.0],
            )
        io.codec_model = None
        io.codec_n_streams = 0
        io.codec_vocab_size = []
        io.codec_sample_rate = None
        io.codec_frame_shift = None
        io.codec_frame_per_second = None
        io.ssl_model = None
        io.km_model = None
        io.ssl_n_streams = 1
        io.ssl_vocab_size = [500]
        io.ssl_sample_rate = 16000
        io.ssl_frame_shift = 320
        io.ssl_frame_per_second = 50
        with pytest.raises(ValueError, match="positive"):
            io._init_sanity_check()


# ---------------------------------------------------------------------------
# DiscreteAudioIO — sanity check
# ---------------------------------------------------------------------------
class TestDiscreteAudioIOSanityCheck:
    def test_mismatched_sample_rate_raises(self):
        with (
            patch.object(DiscreteAudioIO, "_init_codec"),
            patch.object(DiscreteAudioIO, "_init_ssl"),
            patch.object(DiscreteAudioIO, "_init_sanity_check"),
        ):
            io = DiscreteAudioIO(
                ssl_choice="ESPnet",
                ssl_hf_model_tag="espnet/xeus",
                codec_choice="ESPnet",
                codec_hf_model_tag="m",
            )
        io.ssl_model = None
        io.km_model = None
        io.ssl_n_streams = 1
        io.ssl_vocab_size = [500]
        io.ssl_sample_rate = 16000
        io.ssl_frame_shift = 320
        io.ssl_frame_per_second = 50
        io.codec_model = None
        io.codec_n_streams = 2
        io.codec_vocab_size = [1024, 1024]
        io.codec_sample_rate = 24000  # mismatch!
        io.codec_frame_shift = 320
        io.codec_frame_per_second = 50
        with pytest.raises(ValueError, match="Sample rates must match"):
            io._init_sanity_check()

    def test_mismatched_frame_shift_raises(self):
        with (
            patch.object(DiscreteAudioIO, "_init_codec"),
            patch.object(DiscreteAudioIO, "_init_ssl"),
            patch.object(DiscreteAudioIO, "_init_sanity_check"),
        ):
            io = DiscreteAudioIO(
                ssl_choice="ESPnet",
                ssl_hf_model_tag="espnet/xeus",
                codec_choice="ESPnet",
                codec_hf_model_tag="m",
            )
        io.ssl_model = None
        io.km_model = None
        io.ssl_n_streams = 1
        io.ssl_vocab_size = [500]
        io.ssl_sample_rate = 16000
        io.ssl_frame_shift = 320
        io.ssl_frame_per_second = 50
        io.codec_model = None
        io.codec_n_streams = 2
        io.codec_vocab_size = [1024, 1024]
        io.codec_sample_rate = 16000
        io.codec_frame_shift = 480  # mismatch!
        io.codec_frame_per_second = 50
        with pytest.raises(ValueError, match="Frame shifts must match"):
            io._init_sanity_check()


# ---------------------------------------------------------------------------
# DiscreteAudioIO — find_length
# ---------------------------------------------------------------------------
class TestDiscreteAudioIOFindLength:
    def test_basic(self, ssl_only_io):
        wav = np.zeros((1, 32000))  # 2 seconds at 16kHz
        length = ssl_only_io.find_length((wav, 16000))
        assert length == 32000 // 320  # 100 frames

    def test_with_delay_interleave(self, ssl_codec_io):
        ssl_codec_io.delay_interleave = True
        wav = np.zeros((1, 16000))  # 1 second
        length = ssl_codec_io.find_length((wav, 16000))
        base = 16000 // 320  # 50
        expected = base + ssl_codec_io.num_stream() - 1  # 50 + 2 = 52
        assert length == expected

    def test_with_resampling(self, ssl_only_io):
        wav = np.zeros((1, 48000))  # 1s at 48kHz
        length = ssl_only_io.find_length((wav, 48000))
        # 48000 * 16000/48000 = 16000 samples, 16000 // 320 = 50
        assert length == 50


# ---------------------------------------------------------------------------
# DiscreteAudioIO — preprocess
# ---------------------------------------------------------------------------
class TestDiscreteAudioIOPreprocess:
    def test_output_shapes(self, ssl_only_io):
        wav = np.zeros((1, 16000))
        seq, conti_feat, loss_mask = ssl_only_io.preprocess((wav, 16000))
        expected_len = ssl_only_io.find_length((wav, 16000))
        n_streams = ssl_only_io.num_stream()

        assert seq.shape == (expected_len, n_streams)
        assert seq.dtype == np.int32
        assert conti_feat is not None
        assert conti_feat[0] == expected_len
        assert loss_mask.shape == (expected_len, n_streams)

    def test_loss_mask_uses_stream_weights(self, ssl_codec_io):
        ssl_codec_io.stream_weights = [0.5, 1.0, 0.8]
        wav = np.zeros((1, 16000))
        _, _, loss_mask = ssl_codec_io.preprocess((wav, 16000))
        # Each row should match stream_weights
        np.testing.assert_allclose(loss_mask[0], [0.5, 1.0, 0.8])

    def test_conti_feat_contains_audio(self, ssl_only_io):
        wav = np.random.randn(1, 16000).astype(np.float32)
        _, conti_feat, _ = ssl_only_io.preprocess((wav, 16000))
        assert conti_feat[1].shape[1] == 1  # transposed: [samples, channels]


# ---------------------------------------------------------------------------
# DiscreteAudioIO — delay interleave / deinterleave
# ---------------------------------------------------------------------------
class TestDelayInterleave:
    def test_interleave_output_shape(self, ssl_codec_io):
        B, T, N = 2, 10, ssl_codec_io.num_stream()
        codes = torch.randint(0, 100, (B, T, N))
        result = ssl_codec_io._apply_delay_interleave(codes)
        assert result.shape == (B, T + N - 1, N)

    def test_deinterleave_output_shape(self, ssl_codec_io):
        B, T, N = 2, 12, ssl_codec_io.num_stream()
        codes = torch.randint(0, 100, (B, T, N))
        result = ssl_codec_io._apply_delay_deinterleave(codes)
        assert result.shape == (B, T - N + 1, N)

    def test_roundtrip_recovery(self, ssl_codec_io):
        B, T, N = 2, 20, ssl_codec_io.num_stream()
        codes = torch.randint(0, 100, (B, T, N))
        interleaved = ssl_codec_io._apply_delay_interleave(codes)
        recovered = ssl_codec_io._apply_delay_deinterleave(interleaved)
        assert recovered.shape == codes.shape
        torch.testing.assert_close(recovered, codes)

    def test_padding_tokens_in_interleaved(self, ssl_codec_io):
        B, T, N = 1, 5, ssl_codec_io.num_stream()
        # Use values far from pad tokens
        codes = torch.full((B, T, N), 999)
        interleaved = ssl_codec_io._apply_delay_interleave(codes)
        # Stream 1 should have pad at position 0
        pad_tok_stream1 = ssl_codec_io._stream_intervals[1][0]
        assert interleaved[0, 0, 1].item() == pad_tok_stream1
        # Stream 2 should have pad at positions 0 and 1
        pad_tok_stream2 = ssl_codec_io._stream_intervals[2][0]
        assert interleaved[0, 0, 2].item() == pad_tok_stream2
        assert interleaved[0, 1, 2].item() == pad_tok_stream2

    def test_single_stream_interleave_noop(self, ssl_only_io):
        B, T, N = 1, 10, 1
        codes = torch.randint(0, 100, (B, T, N))
        interleaved = ssl_only_io._apply_delay_interleave(codes)
        # With 1 stream, length stays T + 1 - 1 = T
        assert interleaved.shape == (B, T, N)


# ---------------------------------------------------------------------------
# ContinuousAudioIO tests — lightweight logic only (no model loading)
# ---------------------------------------------------------------------------
class TestContinuousAudioIO:
    def _make_continuous_io(self, model_tag="Qwen/Qwen2.5-Omni-7B"):
        with patch.object(ContinuousAudioIO, "_init_encoder"):
            io = ContinuousAudioIO(
                encoder_choice="huggingface",
                encoder_hf_model_tag=model_tag,
            )
        io.d_model = 3584
        io.sample_rate = 16000
        io.hop_length = 160
        io.n_samples = 480000
        return io

    def test_init_attributes(self):
        io = self._make_continuous_io()
        assert io.modality == "audio"
        assert io.is_discrete is False

    def test_feature_dim(self):
        io = self._make_continuous_io()
        assert io.feature_dim() == 3584

    def test_find_length_qwen25(self):
        io = self._make_continuous_io("Qwen/Qwen2.5-Omni-7B")
        # before_length=100: layer1=(100-1)//2+1=50, layer2=(50-2)//2+1=25
        result = io.find_length(None, before_length=100)
        assert result == 25

    def test_find_length_with_tensor(self):
        io = self._make_continuous_io("Qwen/Qwen2.5-Omni-7B")
        lengths = torch.tensor([100, 200, 50])
        result = io.find_length(None, before_length=lengths)
        expected_0 = ((100 - 1) // 2 + 1 - 2) // 2 + 1
        expected_1 = ((200 - 1) // 2 + 1 - 2) // 2 + 1
        expected_2 = ((50 - 1) // 2 + 1 - 2) // 2 + 1
        torch.testing.assert_close(
            result, torch.tensor([expected_0, expected_1, expected_2])
        )

    def test_find_length_unsupported_model_raises(self):
        io = self._make_continuous_io("unknown/model")
        with pytest.raises(NotImplementedError):
            io.find_length(None, before_length=100)
