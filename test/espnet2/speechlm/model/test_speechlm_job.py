"""Tests for espnet2/speechlm/model/speechlm/speechlm_job.py.

Tests SpeechLMJobTemplate and SpeechLMPreprocessor with mock IO classes
that implement the AbsIO interface without heavy dependencies.

``SpeechLMJobTemplate`` transitively imports ``lm/loss.py`` (via
``lm/parallel.py``), which pulls in ``liger_kernel`` at module load
time. Liger is not a declared project dep; the whole file is skipped
(via the sibling ``conftest.py``'s ``collect_ignore_glob``) when
``liger_kernel.ops.fused_linear_cross_entropy`` is not importable.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO


# ---------------------------------------------------------------------------
# Mock IO classes
# ---------------------------------------------------------------------------
class MockDiscreteIO(AbsIO):
    """Text-like discrete IO with 1 stream and 100 tokens."""

    def __init__(self):
        super().__init__(modality="text", is_discrete=True)

    def num_stream(self):
        return 1

    def get_vocabulary(self):
        return [f"tok_{i}" for i in range(100)]

    def get_stream_interval(self):
        return [(0, 100)]

    def get_stream_weight(self):
        return [1.0]

    def preprocess(self, data):
        seq = np.array([[0], [1], [2]], dtype=np.int64)
        loss_mask = np.ones((3, 1), dtype=np.float32)
        return seq, None, loss_mask

    def find_length(self, data):
        return 3

    def copy_for_worker(self):
        return self

    def feature_dim(self):
        return None

    def dummy_forward(self, ref_tensor=None):
        return torch.zeros(1)


class MockDiscreteAudioIO(AbsIO):
    """Audio-like discrete IO with 4 streams and 400 tokens."""

    def __init__(self):
        super().__init__(modality="audio", is_discrete=True)

    def num_stream(self):
        return 4

    def get_vocabulary(self):
        return [f"audio_tok_{i}" for i in range(400)]

    def get_stream_interval(self):
        return [(0, 100), (100, 200), (200, 300), (300, 400)]

    def get_stream_weight(self):
        return [1.0, 1.0, 1.0, 1.0]

    def preprocess(self, data):
        seq = np.zeros((5, 4), dtype=np.int64)
        loss_mask = np.ones((5, 4), dtype=np.float32)
        return seq, None, loss_mask

    def find_length(self, data):
        return 5

    def copy_for_worker(self):
        return self

    def feature_dim(self):
        return None

    def dummy_forward(self, ref_tensor=None):
        return torch.zeros(1)


class MockContinuousAudioIO(AbsIO):
    """Continuous audio IO (feature-based, not tokenized)."""

    def __init__(self):
        super().__init__(modality="audio", is_discrete=False)

    def num_stream(self):
        return None

    def get_vocabulary(self):
        return None

    def get_stream_interval(self):
        return None

    def preprocess(self, data):
        length = 5
        seq = np.zeros((length, 1), dtype=np.int64)
        feat = np.random.randn(length, 80).astype(np.float32)
        conti_feat = (length, torch.from_numpy(feat))
        loss_mask = np.ones((length, 1), dtype=np.float32)
        return seq, conti_feat, loss_mask

    def find_length(self, data):
        return 5

    def copy_for_worker(self):
        return self

    def feature_dim(self):
        return 80

    def dummy_forward(self, ref_tensor=None):
        return torch.zeros(1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
MOCK_IOS = {
    "text": MockDiscreteIO,
    "discrete_audio": MockDiscreteAudioIO,
    "continuous_audio": MockContinuousAudioIO,
}


def _make_config(
    audio_input="continuous_audio",
    audio_output="discrete_audio",
    loss_region="assistant",
    batchfy_method="bucket",
    batch_size=-1,
):
    """Create a minimal SpeechLMJobTemplate config dict."""
    return {
        "multimodal_io": {
            "text": {},
            "discrete_audio": {},
        },
        "preprocessor": {
            "audio_input": audio_input,
            "audio_output": audio_output,
            "loss_region": loss_region,
        },
        "data_loading": {
            "batchfy_method": batchfy_method,
            "batch_size": batch_size,
        },
        "model": {
            "model_choice": "parallel",
            "model_hf_tag": "mock",
            "model_conf": {},
        },
    }


@pytest.fixture
def job_template():
    """Build a SpeechLMJobTemplate with mocked multimodal IOs."""
    with patch(
        "espnet2.speechlm.model.speechlm.speechlm_job._multimodal_ios", MOCK_IOS
    ):
        from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

        return SpeechLMJobTemplate(_make_config(), is_train=True)


@pytest.fixture
def preprocessor(job_template):
    """Build a SpeechLMPreprocessor from the job template."""
    return job_template.build_preprocessor()


@pytest.fixture
def preprocessor_all_loss(job_template):
    """Preprocessor with loss_region='all'."""
    with patch(
        "espnet2.speechlm.model.speechlm.speechlm_job._multimodal_ios", MOCK_IOS
    ):
        from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

        config = _make_config(loss_region="all")
        jt = SpeechLMJobTemplate(config, is_train=True)
        return jt.build_preprocessor()


# ---------------------------------------------------------------------------
# SpeechLMJobTemplate tests
# ---------------------------------------------------------------------------
class TestSpeechLMJobTemplate:
    def test_init_builds_multimodal_io(self, job_template):
        assert isinstance(job_template.multimodal_io, dict)
        assert len(job_template.multimodal_io) > 0
        for name, io in job_template.multimodal_io.items():
            assert isinstance(io, AbsIO)

    def test_init_builds_vocab_meta(self, job_template):
        vm = job_template.vocab_meta
        assert isinstance(vm, dict)
        assert isinstance(vm["vocab"], list)
        assert len(vm["vocab"]) > 0
        assert isinstance(vm["vocab_intervals"], dict)

    def test_vocab_meta_keys(self, job_template):
        """vocab_meta exposes all fields expected by ParallelLLM.from_pretrained."""
        vm = job_template.vocab_meta
        for key in (
            "vocab",
            "vocab_intervals",
            "vocab_weight",
            "vocab_size",
            "mm_start",
            "mm_end",
            "text_start",
            "text_end",
            "num_stream",
        ):
            assert key in vm, f"vocab_meta missing key {key!r}"

    def test_vocab_meta_vocab_size_matches(self, job_template):
        vm = job_template.vocab_meta
        assert vm["vocab_size"] == len(vm["vocab"])

    def test_vocab_weight_shape(self, job_template):
        vm = job_template.vocab_meta
        assert vm["vocab_weight"].shape == (vm["vocab_size"],)
        assert vm["vocab_weight"].dtype == torch.float32

    def test_num_stream_matches_max(self, job_template):
        """num_stream == max stream count among discrete IOs."""
        discrete = [io for io in job_template.multimodal_io.values() if io.is_discrete]
        expected = max(io.num_stream() for io in discrete)
        assert job_template.vocab_meta["num_stream"] == expected

    def test_text_start_end_match_interval(self, job_template):
        vm = job_template.vocab_meta
        iv = vm["vocab_intervals"]["text"]
        assert vm["text_start"] == iv[0][0]
        assert vm["text_end"] == iv[-1][1]

    def test_mm_range_covers_discrete_audio(self, job_template):
        """Default config has text + discrete_audio; mm range spans the audio."""
        vm = job_template.vocab_meta
        audio_iv = vm["vocab_intervals"]["discrete_audio"]
        audio_start = min(s for s, _ in audio_iv)
        audio_end = max(e for _, e in audio_iv)
        assert vm["mm_start"] == audio_start
        assert vm["mm_end"] == audio_end

    def test_build_vocabulary_special_tokens(self, job_template):
        vocab = job_template.vocab_meta["vocab"]
        assert vocab[0] == "<|pad|>"
        assert vocab[1] == "<|bos|>"
        assert vocab[2] == "<|eos|>"
        assert vocab[3] == "<|eot|>"
        for i in range(256):
            assert vocab[i].startswith("<|")

    def test_build_vocabulary_modality_tokens(self, job_template):
        vocab = job_template.vocab_meta["vocab"]
        assert len(vocab) > 256
        assert "tok_0" in vocab

    def test_build_vocabulary_intervals(self, job_template):
        intervals = job_template.vocab_meta["vocab_intervals"]
        assert "special_token" in intervals
        assert intervals["special_token"] == [(0, 256)]
        assert "text" in intervals
        assert intervals["text"][0][0] == 256

    def test_build_vocabulary_no_duplicates(self, job_template):
        vocab = job_template.vocab_meta["vocab"]
        assert len(vocab) == len(set(vocab))

    def test_build_preprocessor_returns_preprocessor(self, job_template):
        from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMPreprocessor

        preprocessor = job_template.build_preprocessor()
        assert isinstance(preprocessor, SpeechLMPreprocessor)

    def test_build_model_dispatches_parallel(self, job_template):
        """Without parallel_dims, default `parallel` class is selected."""
        mock_model = MagicMock()
        mock_class = MagicMock(return_value=mock_model)
        with patch(
            "espnet2.speechlm.model.speechlm.speechlm_job._lms",
            {"parallel": mock_class},
        ):
            job_template.build_model()
            mock_class.assert_called_once()

    def test_preprocessor_no_discrete_raises(self):
        """SpeechLMPreprocessor requires at least one discrete IO."""
        from espnet2.speechlm.model.speechlm.speechlm_job import (
            SpeechLMPreprocessor,
        )

        with pytest.raises(ValueError, match="at least one discrete multimodal IO"):
            SpeechLMPreprocessor(
                is_train=True,
                multimodal_io={"continuous_audio": MockContinuousAudioIO()},
                vocab=["<|pad|>"] + [f"v{i}" for i in range(255)],
                vocab_intervals={"special_token": [(0, 256)]},
            )

    def test_build_model_dispatches_parallel_pp(self, job_template):
        """With parallel_dims.pp_enabled, the `_pp` variant is selected."""
        # build_model wraps PP stages in nn.ModuleList, which rejects
        # MagicMock — return a real nn.Module from the mocked class.
        stage_module = torch.nn.Linear(1, 1)
        mock_parallel = MagicMock(return_value=torch.nn.Linear(1, 1))
        mock_pp = MagicMock(return_value=stage_module)

        # Fake parallel_dims with pp_enabled + a 1-stage pp mesh.
        parallel_dims = MagicMock()
        parallel_dims.pp_enabled = True
        pp_mesh = MagicMock()
        pp_mesh.size.return_value = 1
        parallel_dims.get_mesh.return_value = pp_mesh

        job_template.config["trainer"] = {"titan_config": {"pp_layout": [12]}}

        with patch(
            "espnet2.speechlm.model.speechlm.speechlm_job._lms",
            {"parallel": mock_parallel, "parallel_pp": mock_pp},
        ):
            job_template.build_model(parallel_dims=parallel_dims)
            mock_pp.assert_called_once()
            mock_parallel.assert_not_called()


# ---------------------------------------------------------------------------
# SpeechLMPreprocessor — special_token / special_mask helpers
# ---------------------------------------------------------------------------
class TestSpecialTokenAndMask:
    def test_special_token_shape(self, preprocessor):
        tok = preprocessor.special_token("<|bos|>")
        assert tok.shape == (1, preprocessor.num_stream)
        assert tok.dtype == np.int64

    def test_special_token_first_stream(self, preprocessor):
        tok = preprocessor.special_token("<|bos|>")
        assert tok[0, 0] == preprocessor.vocab.index("<|bos|>")

    def test_special_token_padding(self, preprocessor):
        tok = preprocessor.special_token("<|bos|>")
        # Non-first streams should be pad_id
        for s in range(1, preprocessor.num_stream):
            assert tok[0, s] == preprocessor.pad_id

    def test_special_token_invalid(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor.special_token("<|nonexistent|>")

    def test_special_mask_shape(self, preprocessor):
        m = preprocessor.special_mask(1.0)
        assert m.shape == (1, preprocessor.num_stream)
        assert m.dtype == np.float32

    def test_special_mask_value_zero(self, preprocessor):
        m = preprocessor.special_mask(0.0)
        assert m[0, 0] == 0.0
        for s in range(1, preprocessor.num_stream):
            assert m[0, s] == 0.0

    def test_special_mask_value_one(self, preprocessor):
        m = preprocessor.special_mask(1.0)
        assert m[0, 0] == 1.0
        for s in range(1, preprocessor.num_stream):
            assert m[0, s] == 0.0


# ---------------------------------------------------------------------------
# _apply_chat_template
# ---------------------------------------------------------------------------
class TestApplyChatTemplate:
    def test_chat_template_from_task(self, preprocessor):
        data_dict = {"text1": "hello", "audio1": "/tmp/audio.wav"}
        messages = preprocessor._apply_chat_template("text_to_audio", data_dict)
        assert len(messages) == 2
        # First message: user, text
        assert messages[0][0] == "user"
        assert messages[0][1] == "text"
        # Second message: assistant, audio
        assert messages[1][0] == "assistant"

    def test_chat_template_from_dialogue(self, preprocessor):
        data_dict = {
            "dialogue": [
                ("user", "text", "hello"),
                ("assistant", "text", "hi"),
            ]
        }
        messages = preprocessor._apply_chat_template("text_only", data_dict)
        assert len(messages) == 2
        assert messages[0][0] == "user"
        assert messages[1][0] == "assistant"

    def test_chat_template_dialogue_exclusive(self, preprocessor):
        data_dict = {
            "dialogue": [("user", "text", "hello")],
            "text1": "extra",
        }
        with pytest.raises(ValueError, match="no more other entries"):
            preprocessor._apply_chat_template("text_only", data_dict)

    def test_chat_template_audio_io_selection(self, preprocessor):
        data_dict = {
            "dialogue": [
                ("user", "audio", "/tmp/in.wav"),
                ("assistant", "audio", "/tmp/out.wav"),
            ]
        }
        messages = preprocessor._apply_chat_template("text_to_audio", data_dict)
        # User audio uses audio_input
        assert messages[0][1] == preprocessor.audio_input
        # Assistant audio uses audio_output
        assert messages[1][1] == preprocessor.audio_output

    def test_chat_template_inference_stops_at_assistant(self):
        """When is_train=False, assistant messages should be excluded."""
        with patch(
            "espnet2.speechlm.model.speechlm.speechlm_job._multimodal_ios",
            MOCK_IOS,
        ):
            from espnet2.speechlm.model.speechlm.speechlm_job import (
                SpeechLMJobTemplate,
            )

            config = _make_config()
            jt = SpeechLMJobTemplate(config, is_train=False)
            proc = jt.build_preprocessor()

            data_dict = {
                "dialogue": [
                    ("user", "text", "hello"),
                    ("assistant", "text", "hi"),
                ]
            }
            messages = proc._apply_chat_template("text_only", data_dict)
            # Only user message should be present
            assert len(messages) == 1
            assert messages[0][0] == "user"


# ---------------------------------------------------------------------------
# find_length
# ---------------------------------------------------------------------------
class TestFindLength:
    def test_find_length_text_task(self, preprocessor):
        key = ("text_only", "utt_id", "set_name")
        data_dict = {"text1": "hello"}
        length = preprocessor.find_length(key, data_dict)
        # 1 bos + 1*(role + modality + content_length + eos/eot) = 1 + 3 + 3 = 7
        assert length == 1 + 3 + 3  # MockDiscreteIO.find_length returns 3

    def test_find_length_multi_message(self, preprocessor):
        key = ("text_to_audio", "utt_id", "set_name")
        data_dict = {"text1": "hello", "audio1": "/tmp/audio.wav"}
        length = preprocessor.find_length(key, data_dict)
        # 1 bos + msg1(3+3) + msg2(3+5) = 1 + 6 + 8 = 15
        assert length == 1 + 6 + 8


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def _preprocess_text_only(self, preprocessor):
        key = ("text_only", "utt_id", "set_name")
        data_dict = {"text1": "hello"}
        return preprocessor.preprocessing(key, data_dict)

    def test_preprocessing_returns_expected_keys(self, preprocessor):
        result = self._preprocess_text_only(preprocessor)
        assert "sequence" in result
        assert "conti_feats" in result
        assert "loss_mask" in result

    def test_preprocessing_sequence_shape(self, preprocessor):
        result = self._preprocess_text_only(preprocessor)
        seq = result["sequence"]
        assert seq.ndim == 2
        assert seq.shape[1] == preprocessor.num_stream

    def test_preprocessing_loss_mask_shape(self, preprocessor):
        result = self._preprocess_text_only(preprocessor)
        assert result["sequence"].shape == result["loss_mask"].shape

    def test_preprocessing_starts_with_bos(self, preprocessor):
        result = self._preprocess_text_only(preprocessor)
        bos_id = preprocessor.vocab.index("<|bos|>")
        assert result["sequence"][0, 0] == bos_id

    def test_preprocessing_ends_with_eos(self, preprocessor):
        result = self._preprocess_text_only(preprocessor)
        eos_id = preprocessor.vocab.index("<|eos|>")
        # Last frame, first stream should be eos
        assert result["sequence"][-1, 0] == eos_id

    def test_preprocessing_loss_mask_assistant_only(self, preprocessor):
        """With loss_region='assistant', only assistant region has non-zero loss."""
        key = ("text_to_audio", "utt_id", "set_name")
        data_dict = {"text1": "hello", "audio1": "/tmp/audio.wav"}
        result = preprocessor.preprocessing(key, data_dict)
        loss_mask = result["loss_mask"]

        # BOS should have 0 loss
        assert loss_mask[0, 0] == 0.0

        # The user message region (first message) should have 0 loss
        # user msg: pos 1 (role), 2 (modality), 3-5 (content), 6 (eot)
        # All user tokens should be 0
        for i in range(1, 7):  # positions 1..6 = user message
            assert loss_mask[i, 0] == 0.0

        # Assistant region should have non-zero loss
        # assistant msg starts at 7: role(7), modality(8), content(9..13), eos(14)
        has_nonzero = any(loss_mask[i, 0] > 0 for i in range(7, len(loss_mask)))
        assert has_nonzero

    def test_preprocessing_loss_mask_all(self, preprocessor_all_loss):
        """With loss_region='all', all content regions have non-zero loss."""
        key = ("text_to_audio", "utt_id", "set_name")
        data_dict = {"text1": "hello", "audio1": "/tmp/audio.wav"}
        result = preprocessor_all_loss.preprocessing(key, data_dict)
        loss_mask = result["loss_mask"]
        # Skip BOS (index 0), all other first-stream values should be 1.0
        for i in range(1, len(loss_mask)):
            assert loss_mask[i, 0] == 1.0


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------
class TestCollateFn:
    def _make_sample(self, preprocessor, task="text_only", text="hello"):
        key = (task, "utt_id", "set_name")
        data_dict = {"text1": text}
        return (key, data_dict)

    def test_collate_bucket_basic(self, preprocessor):
        samples = [self._make_sample(preprocessor) for _ in range(2)]
        result = preprocessor.collate_fn(samples)
        assert "seqs" in result
        assert "loss_masks" in result
        assert "keys" in result

    def test_collate_bucket_padding(self, preprocessor):
        samples = [self._make_sample(preprocessor) for _ in range(2)]
        result = preprocessor.collate_fn(samples)
        seqs = result["seqs"]
        # Both samples have same length, so no actual padding difference
        assert seqs.shape[0] == 2

    def test_collate_bucket_seqs_shape(self, preprocessor):
        samples = [self._make_sample(preprocessor) for _ in range(3)]
        result = preprocessor.collate_fn(samples)
        seqs = result["seqs"]
        assert seqs.ndim == 3
        assert seqs.shape[0] == 3
        assert seqs.shape[2] == preprocessor.num_stream

    def test_collate_pack_basic(self):
        """Pack batchfy method concatenates sequences."""
        with patch(
            "espnet2.speechlm.model.speechlm.speechlm_job._multimodal_ios",
            MOCK_IOS,
        ):
            from espnet2.speechlm.model.speechlm.speechlm_job import (
                SpeechLMJobTemplate,
            )

            config = _make_config(batchfy_method="pack")
            jt = SpeechLMJobTemplate(config, is_train=True)
            proc = jt.build_preprocessor()

            key = ("text_only", "utt_id", "set_name")
            samples = [(key, {"text1": "hello"}), (key, {"text1": "world"})]
            result = proc.collate_fn(samples)

            assert "position_ids" in result
            assert result["seqs"].shape[0] == 1  # packed into single batch

    def test_collate_pack_seqs_shape(self):
        with patch(
            "espnet2.speechlm.model.speechlm.speechlm_job._multimodal_ios",
            MOCK_IOS,
        ):
            from espnet2.speechlm.model.speechlm.speechlm_job import (
                SpeechLMJobTemplate,
            )

            config = _make_config(batchfy_method="pack")
            jt = SpeechLMJobTemplate(config, is_train=True)
            proc = jt.build_preprocessor()

            key = ("text_only", "utt_id", "set_name")
            samples = [(key, {"text1": "a"}), (key, {"text1": "b"})]
            result = proc.collate_fn(samples)

            seqs = result["seqs"]
            assert seqs.ndim == 3
            assert seqs.shape[0] == 1  # single packed batch

    def test_collate_pack_batch_length_padding(self):
        with patch(
            "espnet2.speechlm.model.speechlm.speechlm_job._multimodal_ios",
            MOCK_IOS,
        ):
            from espnet2.speechlm.model.speechlm.speechlm_job import (
                SpeechLMJobTemplate,
            )

            config = _make_config(batchfy_method="pack", batch_size=100)
            jt = SpeechLMJobTemplate(config, is_train=True)
            proc = jt.build_preprocessor()

            key = ("text_only", "utt_id", "set_name")
            samples = [(key, {"text1": "a"})]
            result = proc.collate_fn(samples)
            # Should be padded to at least batch_size
            assert result["seqs"].shape[1] >= 100

    def test_collate_invalid_method_raises(self):
        with patch(
            "espnet2.speechlm.model.speechlm.speechlm_job._multimodal_ios",
            MOCK_IOS,
        ):
            from espnet2.speechlm.model.speechlm.speechlm_job import (
                SpeechLMPreprocessor,
            )

            proc = SpeechLMPreprocessor(
                is_train=True,
                multimodal_io={
                    "text": MockDiscreteIO(),
                    "discrete_audio": MockDiscreteAudioIO(),
                },
                vocab=["<|pad|>"]
                + [f"v{i}" for i in range(255)]
                + [f"t{i}" for i in range(100)],
                vocab_intervals={
                    "special_token": [(0, 256)],
                    "text": [(256, 356)],
                },
                audio_input="continuous_audio",
                audio_output="discrete_audio",
            )
            # Force invalid method
            proc.batchfy_method = "invalid"
            key = ("text_only", "utt_id", "set_name")
            with pytest.raises(NotImplementedError):
                proc.collate_fn([(key, {"text1": "hello"})])


# ---------------------------------------------------------------------------
# _apply_cfg
# ---------------------------------------------------------------------------
class TestApplyCfg:
    def test_cfg_no_audio_output(self, preprocessor):
        """Returns unchanged when no assistant audio in messages."""
        messages = [("user", "text", "hello"), ("assistant", "text", "hi")]
        seq = [np.zeros((1, 4))] * (1 + len(messages) * 4)
        loss_masks = [np.zeros((1, 4))] * (1 + len(messages) * 4)
        conti_feats = []
        result_seq, result_lm, result_cf = preprocessor._apply_cfg(
            seq, loss_masks, conti_feats, messages
        )
        # Should be unchanged since no audio output
        assert result_seq is seq
        assert result_lm is loss_masks

    def test_cfg_zeros_non_selected(self, preprocessor):
        """Non-selected segments should be zeroed out."""
        messages = [
            ("user", "text", "hello"),
            ("assistant", preprocessor.audio_output, "/tmp/a.wav"),
        ]
        seq = [np.ones((1, 4))] * (1 + len(messages) * 4)
        loss_masks = [np.ones((1, 4))] * (1 + len(messages) * 4)
        conti_feats = []
        result_seq, result_lm, result_cf = preprocessor._apply_cfg(
            list(seq), list(loss_masks), conti_feats, messages
        )
        # BOS should be zeroed
        assert np.all(result_seq[0] == 0)

    def test_cfg_keeps_one_audio(self, preprocessor):
        """At least one audio output segment should be kept."""
        messages = [
            ("user", "text", "hello"),
            ("assistant", preprocessor.audio_output, "/tmp/a.wav"),
            ("assistant", preprocessor.audio_output, "/tmp/b.wav"),
        ]
        # 1 BOS + 3 messages * 4 items each = 13
        seq = [np.ones((1, 4)) for _ in range(13)]
        loss_masks = [np.ones((1, 4)) for _ in range(13)]
        conti_feats = []
        result_seq, result_lm, _ = preprocessor._apply_cfg(
            seq, loss_masks, conti_feats, messages
        )
        # Exactly one audio segment should be non-zero (4 items per segment)
        # Count non-zero segments (skip BOS at index 0)
        nonzero_segments = 0
        for i in range(len(messages)):
            k = i * 4 + 1  # start of segment
            if np.any(result_seq[k] != 0):
                nonzero_segments += 1
        assert nonzero_segments == 1
