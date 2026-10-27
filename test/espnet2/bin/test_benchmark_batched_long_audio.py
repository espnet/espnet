import torch

from utils.benchmark_batched_long_audio import (
    AcousticSegmenter,
    BatchedLongAudioPipeline,
    DynamicWindowPacker,
    run_benchmark,
)


def test_acoustic_segmenter():
    segmenter = AcousticSegmenter(
        sample_rate=16000,
        max_duration=10.0,
        min_duration=1.0,
    )
    # Audio of 25 seconds duration (should split into at least 3 chunks <= 10s)
    audio = torch.randn(25 * 16000)
    chunks = segmenter.segment("test_utt_01", audio)

    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.audio) <= 10 * 16000
        assert c.utt_id == "test_utt_01"


def test_dynamic_window_packer():
    segmenter = AcousticSegmenter(sample_rate=16000, max_duration=10.0)
    audio = torch.randn(35 * 16000)
    chunks = segmenter.segment("utt_long", audio)

    packer = DynamicWindowPacker(target_batch_size=2)
    micro_batches = packer.pack(chunks)

    assert len(micro_batches) >= 2
    for b in micro_batches:
        assert len(b) <= 2


def test_batched_pipeline_calibrated():
    device = torch.device("cpu")
    segmenter = AcousticSegmenter(sample_rate=16000, max_duration=10.0)
    packer = DynamicWindowPacker(target_batch_size=2)

    pipeline = BatchedLongAudioPipeline(
        segmenter=segmenter,
        packer=packer,
        device=device,
        model_name="synthetic",
    )

    audio = torch.randn(15 * 16000)
    chunks = segmenter.segment("utt_test", audio)
    assert len(chunks) >= 2

    # Test single decode
    text, tokens = pipeline.decode_single(chunks[0].audio)
    assert isinstance(text, str)
    assert len(tokens) > 0

    # Test batch decode
    decoded_chunks = pipeline.decode_batch(chunks)
    assert len(decoded_chunks) == len(chunks)
    for dc in decoded_chunks:
        assert dc.utt_id == "utt_test"
        assert len(dc.text) > 0


def test_run_benchmark_smoke():
    # Smoke test of full end-to-end benchmark in synthetic/calibrated mode
    run_benchmark(
        model_name="synthetic",
        batch_size=2,
        num_rollouts=2,
    )
