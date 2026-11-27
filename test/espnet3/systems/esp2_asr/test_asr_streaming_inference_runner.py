"""Reproduce stateful streaming ASR through the shared inference runner."""

import numpy as np
import pytest

from espnet3.systems.esp2_asr.inference import ASRInferenceRunner

CONTRACT = {
    "inputs": {"speech": {"kind": "audio", "sample_rate": 16000, "channels": 1}},
    "outputs": {"text": {"kind": "text"}},
}


class StreamingBackend:
    """An input-dependent streaming state machine, including an empty prefix."""

    def __init__(self):
        self.parts = []
        self.calls = []
        self.finalizations = 0

    def reset(self):
        self.parts = []

    def __call__(self, speech, is_final=True):
        if isinstance(speech, list):
            raise TypeError("streaming batch unsupported")
        self.calls.append((speech.copy(), is_final))
        self.parts.append(speech.copy())
        wave = np.concatenate(self.parts)
        result = (
            []
            if len(wave) < 4 and not is_final
            else [(f"{len(wave)}:{int(wave.sum())}", [], [], None)]
        )
        if is_final:
            self.finalizations += 1
            self.reset()
        return result


def _forward(backend, chunk, final, idx="utterance"):
    return ASRInferenceRunner.forward(
        0,
        dataset=[{"speech": (16000, chunk), "utt_id": idx}],
        model=backend,
        input_key="speech",
        inference_contract=CONTRACT,
        model_kwargs={"is_final": final},
    )


@pytest.mark.parametrize("length", [3, 4, 5, 8, 9])
def test_streaming_chunks_match_direct_backend_and_flush_once(length):
    """Tail and exact-multiple lengths preserve every sample and final flag."""
    wave = np.arange(length, dtype=np.float32)
    direct, wrapped = StreamingBackend(), StreamingBackend()
    for start in range(0, length, 4):
        chunk = wave[start : start + 4]
        native = direct(chunk, is_final=False)
        result = _forward(wrapped, chunk, False)
        assert result == {"text": native[0][0] if native else "", "utt_id": "utterance"}
    native = direct(np.empty(0, dtype=np.float32), is_final=True)
    assert (
        _forward(wrapped, np.empty(0, dtype=np.float32), True)["text"] == native[0][0]
    )
    assert wrapped.finalizations == direct.finalizations == 1
    assert wrapped.parts == []
    assert len(wrapped.calls) == len(direct.calls)
    for (actual, actual_final), (expected, expected_final) in zip(
        wrapped.calls, direct.calls
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual_final is expected_final


def test_streaming_empty_prefix_and_cumulative_hypotheses():
    """No-hypothesis chunks are valid and partial snapshots are not appended."""
    backend = StreamingBackend()
    assert _forward(backend, np.ones(1, dtype=np.float32), False)["text"] == ""
    assert _forward(backend, np.ones(3, dtype=np.float32), False)["text"] == "4:4"
    assert _forward(backend, np.ones(2, dtype=np.float32), True)["text"] == "6:6"


def test_streaming_finalization_and_explicit_reset_isolate_utterances():
    """A later utterance matches a new decoder, including after cancellation."""
    backend = StreamingBackend()
    _forward(backend, np.ones(8, dtype=np.float32), True, "a")
    wave = np.full(6, 2, dtype=np.float32)
    assert _forward(backend, wave, True, "b") == _forward(
        StreamingBackend(), wave, True, "b"
    )
    _forward(backend, np.ones(8, dtype=np.float32), False)
    backend.reset()
    assert _forward(backend, wave, True, "b") == _forward(
        StreamingBackend(), wave, True, "b"
    )


def test_streaming_sessions_do_not_share_state():
    """Interleaved sessions retain their own sample histories."""
    a, b = StreamingBackend(), StreamingBackend()
    _forward(a, np.ones(3, dtype=np.float32), False)
    _forward(b, np.full(5, 2, dtype=np.float32), False)
    assert _forward(a, np.ones(2, dtype=np.float32), True)["text"] == "5:5"
    assert _forward(b, np.full(2, 2, dtype=np.float32), True)["text"] == "7:14"


def test_streaming_invalid_chunk_does_not_advance_decoder():
    """Rejected audio leaves the decoder's existing utterance state intact."""
    backend = StreamingBackend()
    _forward(backend, np.ones(3, dtype=np.float32), False)
    with pytest.raises(ValueError, match="finite"):
        _forward(backend, np.array([np.nan], dtype=np.float32), False)
    assert len(backend.calls) == 1
    assert _forward(backend, np.ones(2, dtype=np.float32), True)["text"] == "5:5"


def test_streaming_rejects_batch_without_retry():
    """Unsupported multi-stream batching must not become sequential decoding."""
    backend = StreamingBackend()
    with pytest.raises(TypeError, match="batch unsupported"):
        ASRInferenceRunner.forward(
            [0, 1],
            dataset=[{"speech": np.ones(16)}, {"speech": np.ones(32)}],
            model=backend,
            input_key="speech",
            inference_contract=CONTRACT,
        )
    assert backend.calls == []


def test_streaming_backend_failure_propagates_and_explicit_reset_recovers():
    class FailingBackend(StreamingBackend):
        fail = True

        def __call__(self, speech, is_final=True):
            result = super().__call__(speech, is_final=is_final)
            if self.fail:
                raise RuntimeError("streaming decoder failed")
            return result

    backend = FailingBackend()
    with pytest.raises(RuntimeError, match="streaming decoder failed"):
        _forward(backend, np.ones(3, dtype=np.float32), False)
    # Recovery after a decoder error requires an explicit reset; the runner
    # does not promise to roll back native streaming caches.
    backend.reset()
    backend.fail = False
    wave = np.ones(4, dtype=np.float32)
    assert _forward(backend, wave, True) == _forward(StreamingBackend(), wave, True)


@pytest.mark.execution_timeout(90)
def test_real_streaming_decoder_matches_runner_with_identical_chunks(tmp_path):
    """A tiny native decoder exercises the actual frontend and cache lifecycle."""
    import torch

    from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
    from espnet2.tasks.asr import ASRTask

    torch.manual_seed(0)
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blank>\n<unk>\na\nb\n<sos/eos>\n")
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "model"),
            "--token_list",
            str(tokens),
            "--token_type",
            "char",
            "--encoder",
            "contextual_block_transformer",
            "--encoder_conf",
            "{output_size: 8, attention_heads: 2, linear_units: 8, num_blocks: 1, "
            "block_size: 40, hop_size: 16, look_ahead: 16}",
            "--decoder",
            "transformer",
            "--decoder_conf",
            "{attention_heads: 2, linear_units: 8, num_blocks: 1}",
            "--frontend",
            "default",
            "--frontend_conf",
            "{n_fft: 256, win_length: 256, hop_length: 128}",
        ]
    )
    kwargs = {"asr_train_config": tmp_path / "model" / "config.yaml", "beam_size": 1}
    direct = Speech2TextStreaming(**kwargs)
    wrapped = Speech2TextStreaming(**kwargs)
    wrapped.asr_model.load_state_dict(direct.asr_model.state_dict())
    wave = np.random.default_rng(0).normal(size=10240).astype(np.float32)
    expected = direct(wave, is_final=True)
    actual = _forward(wrapped, wave, True, "whole")
    assert actual["text"] == (expected[0][0] or "" if expected else "")
    for utterance in ("a", "b"):
        for start in range(0, len(wave), 2048):
            chunk = wave[start : start + 2048]
            expected = direct(chunk, is_final=False)
            actual = _forward(wrapped, chunk, False, utterance)
            assert actual["text"] == (expected[0][0] or "" if expected else "")
            assert actual["utt_id"] == utterance
        empty = np.empty(0, dtype=np.float32)
        expected = direct(empty, is_final=True)
        actual = _forward(wrapped, empty, True, utterance)
        assert actual["text"] == (expected[0][0] or "" if expected else "")
        assert wrapped.frontend_states is None
        assert wrapped.encoder_states is None
