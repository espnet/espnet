#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Generate structured reasoning traces from user requests to rich captions."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Version-specific prompts for reasoning trace generation
# Add new versions here as needed
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are an AI assistant helping generate audio. Your task is to "
            "produce a structured reasoning trace that bridges a user's request "
            "to a detailed audio description.\n\n"
            "Your reasoning MUST follow this exact structure:\n"
            "1. User Intent: What the user explicitly requested\n"
            "2. Inferred Details: Reasonable details to fill in gaps not "
            "specified by the user\n"
            "3. Audio Quality: Specify the audio quality to generate, including "
            "recording quality (e.g., studio, amateur, lo-fi), clarity, "
            "and any acoustic characteristics\n"
            "4. Generation Plan: Brief summary of how to create this audio\n\n"
            "Output ONLY the structured reasoning trace. "
            "Do NOT include the target description."
        ),
        "user": (
            "Given a user's audio generation request, produce a structured "
            "reasoning trace that leads to the target audio description.\n\n"
            "User request:\n{user_request}\n\n"
            "Target description (for reference, do NOT include in output):\n"
            "{qwen_caption}\n\n"
            "Generate the structured reasoning trace:"
        ),
    },
    # Add more versions below as needed:
    # "v2": {
    #     "system": "...",
    #     "user": "...",
    # },
}

def get_prompts(version: str) -> Dict[str, str]:
    """Get prompts for a specific version.

    Raises:
        ValueError: If the requested version is not found.
    """
    if version not in PROMPTS_BY_VERSION:
        available = list(PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available versions: {available}"
        )
    return PROMPTS_BY_VERSION[version]


def validate_reasoning_trace(trace: str) -> bool:
    """Validate that reasoning trace contains required sections."""
    if trace is None or len(trace.strip()) < 50:
        return False

    required_markers = [
        "User Intent",
        "Inferred Details",
        "Audio Quality",
        "Generation Plan",
    ]

    trace_lower = trace.lower()
    found_count = sum(
        1 for marker in required_markers
        if marker.lower() in trace_lower
    )

    # Require at least 3 of 4 markers to be flexible
    return found_count >= 3


def build_reasoning_query(
    sample: Dict[str, Any],
    detail_level: str,
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for reasoning trace generation."""
    user_requests = sample.get("user_requests", {})
    user_request = user_requests.get(detail_level, "")
    qwen_caption = sample.get("qwen_caption", "")

    if not user_request or not qwen_caption:
        return None

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                user_request=user_request,
                qwen_caption=qwen_caption,
            ),
        },
    ]

    return {
        'idx': sample.get("idx"),
        'messages': messages,
        'temperature': 0.3,
        'max_tokens': 1024,
        'metadata': {
            'sample': sample,
            'detail_level': detail_level,
            'user_request': user_request,
        },
    }


def process_reasoning_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a reasoning trace generation result."""
    response = result.get('response')
    metadata = result.get('metadata', {})
    sample = metadata.get('sample', {})

    if not validate_reasoning_trace(response):
        return None

    return {
        "idx": result['idx'],
        "dataset": sample.get("dataset", ""),
        "audio_path": sample.get("audio_path", ""),
        "qwen_caption": sample.get("qwen_caption", ""),
        "user_request": metadata.get('user_request', ""),
        "reasoning_trace": response.strip(),
        "detail_level": metadata.get('detail_level', ""),
        "persona": sample.get("persona", ""),
    }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input JSONL file with user requests."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


async def main_async(args):
    """Main async function."""
    # Get version-specific prompts
    prompts = get_prompts(args.version)
    print(f"Using prompt version: {args.version}")

    # Load input data
    print(f"Loading input data from {args.input_file}...")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    # Limit samples if specified
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
        print(f"Limited to {len(samples)} samples for processing")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"reasoning_{args.detail_level}.jsonl"
    )

    # Check for existing progress using index-based tracking
    processed_indices = set()
    if os.path.exists(output_file) and args.resume:
        processed_indices = get_processed_indices(output_file)
        print(f"Resuming: found {len(processed_indices)} already processed samples")
    else:
        # Clear output file
        open(output_file, "w").close()

    # Parse URLs and initialize processor
    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    num_servers = len(base_urls)
    workers_per_server = max(1, args.num_workers // num_servers)

    # Build queries for all pending samples
    queries = []
    for sample in samples:
        idx = sample.get("idx")
        if idx in processed_indices:
            continue
        query = build_reasoning_query(sample, args.detail_level, prompts)
        if query is not None:
            queries.append(query)

    print(f"Queries to process: {len(queries)}")

    if not queries:
        print("All samples already processed!")
        return

    # Process all queries
    processor = MassiveQueryProcessor(
        base_urls=base_urls,
        model=model,
        workers_per_server=workers_per_server,
        timeout=args.timeout,
        checkpoint_interval=10000,
    )

    await processor.process_all(
        queries=queries,
        output_file=output_file,
        process_fn=process_reasoning_result,
    )

    # Final summary
    final_count = len(get_processed_indices(output_file))
    print(f"\nDone! Total success: {final_count}/{len(samples)}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces from user requests using vLLM."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with user requests.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for reasoning traces.",
    )
    parser.add_argument(
        "--detail_level",
        type=str,
        required=True,
        choices=["realistic", "imaginary"],
        help="Detail level to process (realistic/imaginary).",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for vLLM API. Defaults to shared module default.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=256,
        help="Number of concurrent API requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Prompt version to use (default: v1).",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
