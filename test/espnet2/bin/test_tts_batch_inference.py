"""Unit tests for TTS batch inference functionality.

This test module validates the batch_call function for TTS
inference, ensuring that batch inference produces identical results
to sequential single-sample inference.

To run this specific test:
    pytest test/espnet2/bin/test_tts_batch_inference.py -v

To run only the main comparison test:
    pytest test/espnet2/bin/test_tts_batch_inference.py::
        test_batch_call_vs_single_call -v

To run with verbose output showing shape comparisons:
    pytest test/espnet2/bin/test_tts_batch_inference.py::
        test_batch_call_vs_single_call -v -s
"""

import string
from pathlib import Path

import pytest
import torch

from espnet2.bin.tts_inference import Text2Speech
from espnet2.tasks.tts import TTSTask


@pytest.fixture()
def token_list(tmp_path: Path):
    """Create a token list file."""
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def fastspeech2_config_file(tmp_path: Path, token_list):
    """Create a FastSpeech2 configuration file for batch inference.

    Args:
        tmp_path: Temporary directory path.
        token_list: Path to token list file.

    Returns:
        Path to the FastSpeech2 configuration file.
    """
    TTSTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "fs2"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--cleaner",
            "none",
            "--g2p",
            "none",
            "--normalize",
            "none",
            "--tts",
            "fastspeech2",
            "--pitch_extract",
            "dio",
            "--energy_extract",
            "energy",
        ]
    )
    return tmp_path / "fs2" / "config.yaml"


@pytest.mark.execution_timeout(20)
def test_batch_call_vs_single_call(fastspeech2_config_file):
    """Test that batch_call produces same output as multiple calls.

    This test validates that:
    1. batch_call handles multiple sentences correctly
    2. The output from batch_call matches the output from individual
       calls
    3. No regression occurs in batch inference functionality
    """
    # Initialize Text2Speech with FastSpeech2
    text2speech = Text2Speech(
        train_config=fastspeech2_config_file,
        always_fix_seed=True,
        seed=777,
    )

    # Check if the model supports batch inference
    if not hasattr(text2speech.model, "batch_inference"):
        pytest.skip("Model does not support batch inference")

    # Test sentences
    test_sentences = ["hello", "world", "test"]

    # ===== Run batch inference =====
    batch_output = text2speech.batch_call(test_sentences)

    # Verify batch output structure
    assert "feat_gen" in batch_output, "feat_gen should be in batch output"
    assert isinstance(batch_output["feat_gen"], list), "feat_gen should be a list"
    assert len(batch_output["feat_gen"]) == len(test_sentences), (
        f"Expected {len(test_sentences)} outputs, "
        f"got {len(batch_output['feat_gen'])}"
    )

    # ===== Run single inference for each sentence =====
    single_outputs = []
    for sentence in test_sentences:
        single_output = text2speech(sentence)
        single_outputs.append(single_output)

    # ===== Compare outputs =====
    print("\n" + "=" * 60)
    print("Comparing batch_call vs single call outputs:")
    print("=" * 60)

    for i, (batch_feat, single_output) in enumerate(
        zip(batch_output["feat_gen"], single_outputs)
    ):
        single_feat = single_output["feat_gen"]

        print(f"\nSentence {i + 1}: '{test_sentences[i]}'")
        print(f"  Batch output shape:  {batch_feat.shape}")
        print(f"  Single output shape: {single_feat.shape}")

        # Check that shapes match
        assert batch_feat.shape == single_feat.shape, (
            f"Shape mismatch for sentence {i}: "
            f"batch={batch_feat.shape} vs single={single_feat.shape}"
        )

        # Check that values are close (allowing for small numerical
        # differences)
        max_diff = torch.abs(batch_feat - single_feat).max().item()
        mean_diff = torch.abs(batch_feat - single_feat).mean().item()

        print(f"  Max difference:      {max_diff:.6f}")
        print(f"  Mean difference:     {mean_diff:.6f}")

        # Use a tolerance for floating point comparison
        assert torch.allclose(batch_feat, single_feat, rtol=1e-2, atol=1e-2), (
            f"Output mismatch for sentence {i}: "
            f"max_diff={max_diff}, mean_diff={mean_diff}"
        )

        print("  ✓ Outputs match!")


@pytest.mark.execution_timeout(10)
def test_batch_call_with_different_lengths(fastspeech2_config_file):
    """Test batch_call with sentences of varying lengths."""
    text2speech = Text2Speech(
        train_config=fastspeech2_config_file,
        always_fix_seed=True,
        seed=777,
    )

    if not hasattr(text2speech.model, "batch_inference"):
        pytest.skip("Model does not support batch inference")

    # Test with sentences of different lengths
    test_sentences = ["a", "hello world", "this is a longer test sentence"]

    # Run batch inference
    batch_output = text2speech.batch_call(test_sentences)

    # Verify outputs
    assert "feat_gen" in batch_output
    assert len(batch_output["feat_gen"]) == len(test_sentences)

    # Check that longer sentences produce longer outputs
    lengths = [feat.size(0) for feat in batch_output["feat_gen"]]
    print(f"\nOutput lengths for different input lengths: {lengths}")

    # Generally, longer input should produce longer output
    # (though not strictly monotonic due to phoneme mapping)
    assert (
        lengths[0] < lengths[2]
    ), "Shortest input should produce shorter output than longest"

    print("✓ Variable length batch inference test passed!")
