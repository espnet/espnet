#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Assemble SFT dialogue from user requests, reasoning traces, and audio."""

import argparse
import json
import os
from typing import Any, Dict, List

# =============================================================================
# Version-specific system prompts for dialogue assembly
# Add new versions here as needed
# =============================================================================
SYSTEM_PROMPTS_BY_VERSION = {
    "v1": (
        "You are a helpful assistant that generates audio based on user "
        "requests. You can create various types of audio including sound "
        "effects, music, speech, ambient sounds, and any combination of these. "
        "When given a request, first think through what the user wants and how "
        "to create high-quality audio, then provide a detailed description of "
        "the audio you will generate."
    ),
    # Add more versions below as needed:
    # "v2": "...",
}

def get_system_prompt(version: str) -> str:
    """Get system prompt for a specific version.

    Raises:
        ValueError: If the requested version is not found.
    """
    if version not in SYSTEM_PROMPTS_BY_VERSION:
        available = list(SYSTEM_PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available versions: {available}"
        )
    return SYSTEM_PROMPTS_BY_VERSION[version]


def create_example_id(sample: Dict[str, Any], detail_level: str) -> str:
    """Create a unique example ID."""
    dataset = sample.get("dataset", "unknown")
    idx = sample.get("idx", 0)
    return f"{dataset}_{idx}_{detail_level}"


def format_assistant_text(
    reasoning_trace: str,
    rich_caption: str,
) -> str:
    """Format assistant text with thinking tokens."""
    # Wrap reasoning trace with <think> tokens
    text = f"<think>\n{reasoning_trace.strip()}\n</think>\n\n{rich_caption.strip()}"
    return text


def create_dialogue(
    sample: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, Any]:
    """Create dialogue format from sample.

    Args:
        sample: Input sample with user_request, reasoning_trace, etc.
        system_prompt: Version-specific system prompt to use.

    Returns:
        Dialogue dict with messages and metadata.
    """
    user_request = sample.get("user_request", "")
    reasoning_trace = sample.get("reasoning_trace", "")
    qwen_caption = sample.get("qwen_caption", "")
    audio_path = sample.get("audio_path", "")
    detail_level = sample.get("detail_level", "short")

    # Create messages in the expected format: [(role, modality, content), ...]
    messages = [
        ["system", "text", system_prompt],
        ["user", "text", user_request],
        [
            "assistant",
            "text",
            format_assistant_text(reasoning_trace, qwen_caption),
        ],
        ["assistant", "audio", audio_path],
    ]

    return {
        "example_id": create_example_id(sample, detail_level),
        "messages": messages,
        "metadata": {
            "dataset": sample.get("dataset", ""),
            "detail_level": detail_level,
            "persona": sample.get("persona", ""),
            "original_idx": sample.get("idx"),
        },
    }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input JSONL file."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Validate that sample has all required fields."""
    required_fields = [
        "user_request",
        "reasoning_trace",
        "qwen_caption",
        "audio_path",
    ]
    return all(
        sample.get(field) and len(str(sample.get(field)).strip()) > 0
        for field in required_fields
    )


def main():
    parser = argparse.ArgumentParser(
        description="Assemble SFT dialogues from reasoning traces."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with reasoning traces.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for assembled dialogues.",
    )
    parser.add_argument(
        "--detail_level",
        type=str,
        required=True,
        choices=["realistic", "imaginary"],
        help="Detail level being processed.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="System prompt version to use (default: v1).",
    )
    args = parser.parse_args()

    # Get version-specific system prompt
    system_prompt = get_system_prompt(args.version)
    print(f"Using system prompt version: {args.version}")

    # Load input data
    print(f"Loading input data from {args.input_file}...")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"dialogues_{args.detail_level}.jsonl"
    )

    # Process samples
    valid_count = 0
    invalid_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            if not validate_sample(sample):
                invalid_count += 1
                continue

            dialogue = create_dialogue(sample, system_prompt)
            f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"Done! Valid: {valid_count}, Invalid: {invalid_count}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
