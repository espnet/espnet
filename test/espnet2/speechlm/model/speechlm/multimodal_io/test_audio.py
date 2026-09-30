"""Tests for KmeansModel, DiscreteAudioIO, and ContinuousAudioIO."""

import types
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

    def test_worker_copy_keeps_original_models_and_preprocessing(self, codec_only_io):
        io = codec_only_io
        io.codec_model = torch.nn.Linear(2, 2)
        original_model = io.codec_model
        with patch.object(DiscreteAudioIO, "_init_codec", side_effect=AssertionError):
            worker = io.copy_for_worker()
        assert io.codec_model is original_model
        assert list(io.parameters())
        assert not list(worker.parameters())
        assert worker.get_vocabulary() == io.get_vocabulary()
        sample = (np.zeros((1, 1600), dtype=np.float32), 16000)
        expected = io.preprocess(sample)
        actual = worker.preprocess(sample)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[2], expected[2])
        assert actual[1][0] == expected[1][0]
        torch.testing.assert_close(actual[1][1], expected[1][1])


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

    def test_worker_copy_keeps_original_encoder(self):
        io = self._make_continuous_io()
        io.model = torch.nn.Linear(2, 2)
        original_model = io.model
        with patch.object(
            ContinuousAudioIO, "_init_encoder", side_effect=AssertionError
        ):
            worker = io.copy_for_worker()
        assert io.model is original_model
        assert list(io.parameters())
        assert not list(worker.parameters())
        sample = (np.zeros((1, 1600), dtype=np.float32), 16000)
        assert worker.find_length(sample) == io.find_length(sample)
        assert worker.feature_dim() == io.feature_dim()

    def test_feature_dim(self):
        io = self._make_continuous_io()
        assert io.feature_dim() == 3584

    @pytest.mark.parametrize(
        "model_tag", ["Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen3-Omni-30B-A3B-Instruct"]
    )
    @pytest.mark.parametrize("structured_output", [False, True])
    def test_encode_batch_extracts_features(self, model_tag, structured_output):
        """Preserve features from tensor and structured encoder outputs."""
        io = self._make_continuous_io(model_tag)
        lengths = torch.tensor([100, 50])
        output_lengths = io.find_length(None, before_length=lengths)
        features = torch.arange(int(output_lengths.sum()) * 4).reshape(-1, 4)
        batch = torch.randn(2, 100, 80)

        class Encoder(torch.nn.Module):
            def get_audio_features(self, data, feature_attention_mask):
                torch.testing.assert_close(data, batch.transpose(1, 2))
                expected = torch.arange(100).unsqueeze(0) < lengths.unsqueeze(1)
                torch.testing.assert_close(feature_attention_mask, expected.int())
                return (
                    types.SimpleNamespace(last_hidden_state=features)
                    if structured_output
                    else features
                )

        io.model = Encoder()
        result = io.encode_batch(batch, lengths)
        assert [value.shape[0] for value in result] == output_lengths.tolist()
        torch.testing.assert_close(torch.cat(result), features)

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

    def test_find_length_from_data_resamples(self):
        io = self._make_continuous_io("Qwen/Qwen2.5-Omni-7B")
        # 48k samples at 48kHz -> 16k samples at 16kHz -> 100 frames -> 25 after ds
        wav = np.zeros((1, 48000))
        assert io.find_length((wav, 48000)) == 25

    def test_find_length_from_data_matches_before_length_path(self):
        io = self._make_continuous_io("Qwen/Qwen2.5-Omni-7B")
        wav = np.zeros((1, 16000))
        assert io.find_length((wav, 16000)) == io.find_length(None, before_length=100)

    def test_preprocess_transposes_TC_input(self):
        from unittest.mock import MagicMock

        io = self._make_continuous_io("Qwen/Qwen2.5-Omni-7B")
        T, feat_dim = 16000, 80
        before_length = T // io.hop_length  # 100
        after_length = io.find_length(None, before_length=before_length)  # 25

        # Processor returns a dict-like with input_features [1, feat_dim, T]
        io.processor = MagicMock(
            return_value={
                "input_features": np.zeros((1, feat_dim, T), dtype=np.float32)
            }
        )

        wav_tc = np.zeros((T, 1), dtype=np.float32)  # [T, C] with T > C
        seq, conti_feat, loss_mask = io.preprocess((wav_tc, io.sample_rate))

        assert seq.shape == (after_length, 1)
        assert conti_feat[0] == after_length
        assert conti_feat[1].shape == (before_length, feat_dim)
        assert loss_mask.shape == (after_length, 1)


# ---------------------------------------------------------------------------
# What _init_encoder asks transformers for. The encoder itself is 30B, so the
# two `from_pretrained` calls are replaced and only the asking is checked.
# ---------------------------------------------------------------------------
class TestContinuousAudioIOEncoderLoading:
    TAG = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    def _fake_omni(self):
        """A stand-in for the checkpoint, with the parts _init_encoder drops."""
        thinker = types.SimpleNamespace(
            model=object(),
            visual=object(),
            lm_head=object(),
            audio_tower=types.SimpleNamespace(
                config=types.SimpleNamespace(output_dim=2048)
            ),
        )
        thinker.to = lambda device: thinker
        return types.SimpleNamespace(thinker=thinker)

    def test_the_feature_extractor_comes_without_the_video_one(self):
        """Ask for the audio feature extractor, not the omni processor.

        An omni processor also builds the video processor, which imports
        torchvision - a package no espnet extra declares and this
        speech-only path never uses. AutoFeatureExtractor reads the same
        `feature_extractor_type` from the same config.
        """
        import transformers

        fake = self._fake_omni()
        extractor = types.SimpleNamespace(sampling_rate=16000, hop_length=160)
        asked = {}

        def feature_extractor(tag, *args, **kwargs):
            asked["tag"] = tag
            return extractor

        def model_class(tag, **kwargs):
            asked["model_tag"] = tag
            return fake

        with (
            patch.object(
                transformers.AutoFeatureExtractor, "from_pretrained", feature_extractor
            ),
            patch.object(
                transformers.Qwen3OmniMoeForConditionalGeneration,
                "from_pretrained",
                model_class,
            ),
            patch.object(transformers.AutoProcessor, "from_pretrained", _never_called),
        ):
            io = ContinuousAudioIO(
                encoder_choice="huggingface",
                encoder_hf_model_tag=self.TAG,
            )

        assert asked["tag"] == self.TAG
        assert io.processor is extractor
        # and the attributes the rest of the class reads off it
        assert io.sample_rate == 16000
        assert io.hop_length == 160
        assert io.d_model == 2048


def _never_called(*args, **kwargs):  # pragma: no cover - the point is that it is not
    raise AssertionError(
        "AutoProcessor builds the image and video processors too; the audio "
        "path needs AutoFeatureExtractor"
    )
