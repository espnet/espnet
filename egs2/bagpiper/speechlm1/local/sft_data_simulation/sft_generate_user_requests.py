#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Generate diverse user requests from rich captions using vLLM API."""

import argparse
import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional

# Fixed random seed for reproducibility when shuffling samples
RANDOM_SEED = 42

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# Persona pool for diversity
PERSONA_POOL = [
    "a content creator making YouTube videos",
    "a musician composing for a new album",
    "a sound designer for video games",
    "a podcaster creating intro/outro sounds",
    "a game developer building an indie game",
    "a film maker working on a short film",
    "an advertising professional making commercials",
    "a casual user exploring audio generation",
    "an educator creating learning materials",
    "an animator working on a cartoon",
    "a meditation app developer",
    "a radio producer",
    "a theater sound technician",
    "a mobile app developer adding sound effects",
    "a documentary filmmaker",
    "a DJ creating sound samples",
    "a voice-over artist",
    "an audiobook producer",
    "a museum exhibit designer",
    "a VR/AR experience developer",
]

# =============================================================================
# Version-specific prompts for user request generation
# Add new versions here as needed
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "persona_selection_system": (
            "You are helping select a user persona for generating audio "
            "requests. Given an audio description, randomly select ONE persona "
            "from the provided list that could reasonably need this audio.\n\n"
            "IMPORTANT: Many personas can use various audio types. For example, "
            "a film maker, game developer, or content creator could all need "
            "music, speech, or sound effects. Only exclude personas where "
            "there is a clear mismatch (e.g., 'voice-over artist' for "
            "non-speech audio).\n\n"
            "To ensure diversity, do NOT always pick the most obvious choice. "
            "Pick randomly among all compatible personas.\n\n"
            "Output ONLY the exact persona string from the list. "
            "No explanation or additional text."
        ),
        "persona_selection_user": (
            "Audio description:\n{qwen_caption}\n\n"
            "Available personas:\n{persona_list}\n\n"
            "Randomly select ONE compatible persona for this audio:"
        ),
        "system": (
            "You simulate users requesting audio generation. "
            "Output ONLY valid JSON, no other text."
        ),
        "user": (
            "You are {persona}. Generate 2 user requests for this audio:\n\n"
            "Rich caption:\n{qwen_caption}\n\n"
            "Generate these 2 request types:\n"
            "1. REALISTIC: Directly describe the audio events, sounds, and "
            "details the user wants.\n"
            "2. IMAGINARY: Describe a SCENE, FEELING, ATMOSPHERE, or abstract "
            "concept that evokes this audio - do NOT mention audio/sound "
            "directly. Let the audio be inferred from the description.\n\n"
            "RULES:\n"
            "- HUMAN-LIKE: Write as a real person would. AVOID technical audio "
            "terms (e.g., 'frequency response', 'stereo panning', 'sidechain', "
            "'compression', 'reverb tail', 'EQ'). Describe WHAT you want, not "
            "HOW to produce it technically.\n"
            "- SPEECH/LYRICS: If the caption contains spoken words or lyrics, "
            "include the EXACT text in BOTH requests. Quoted speech/lyrics do "
            "NOT count toward the word limit.\n"
            "- LENGTH: 15-80 words (excluding quoted speech/lyrics). Vary "
            "naturally - sometimes brief, sometimes detailed, regardless of "
            "audio complexity. Diversity is key.\n"
            "- TONE: Match the persona's style - casual, professional, "
            "enthusiastic, or neutral.\n"
            "- IMAGINARY EXAMPLE: Instead of 'forest ambient sounds', write "
            "'Walking through an ancient forest at dawn, mist rising from the "
            "ground, birds just starting to wake.'\n\n"
            'Output JSON: {{"realistic": "...", "imaginary": "..."}}'
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


def parse_json_response(response: str) -> Optional[Dict[str, str]]:
    """Parse JSON response, handling potential formatting issues."""
    if response is None:
        return None

    required_keys = ["realistic", "imaginary"]

    # Try direct parsing first
    try:
        data = json.loads(response)
        if all(k in data for k in required_keys):
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from response
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            if all(k in data for k in required_keys):
                return data
    except json.JSONDecodeError:
        pass

    return None


def parse_persona_response(response: str) -> Optional[str]:
    """Parse persona selection response and validate against pool."""
    if response is None:
        return None

    response = response.strip()

    # Check for exact match
    if response in PERSONA_POOL:
        return response

    # Check for partial match (LLM might add/remove articles or punctuation)
    response_lower = response.lower()
    for persona in PERSONA_POOL:
        if persona.lower() in response_lower or response_lower in persona.lower():
            return persona

    # No valid match found - sample will be skipped
    return None


def build_persona_query(
    idx: int,
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for persona selection."""
    qwen_caption = sample.get("qwen_caption", "")
    if not qwen_caption or len(qwen_caption.strip()) < 50:
        return None

    persona_list = "\n".join(f"- {p}" for p in PERSONA_POOL)

    messages = [
        {"role": "system", "content": prompts["persona_selection_system"]},
        {
            "role": "user",
            "content": prompts["persona_selection_user"].format(
                qwen_caption=qwen_caption,
                persona_list=persona_list,
            ),
        },
    ]

    return {
        'idx': idx,
        'messages': messages,
        'temperature': 0.3,
        'max_tokens': 64,
        'metadata': sample,
    }


def build_request_query(
    idx: int,
    sample: Dict[str, Any],
    persona: str,
    prompts: Dict[str, str],
) -> Dict[str, Any]:
    """Build a query for user request generation."""
    qwen_caption = sample.get("qwen_caption", "")

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                persona=persona,
                qwen_caption=qwen_caption,
            ),
        },
    ]

    return {
        'idx': idx,
        'messages': messages,
        'temperature': 0.7,
        'max_tokens': 512,
        'metadata': {**sample, 'persona': persona},
    }


def process_request_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a user request generation result."""
    response = result.get('response')
    metadata = result.get('metadata', {})

    parsed = parse_json_response(response)
    if parsed is None:
        return None

    return {
        "idx": result['idx'],
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "qwen_caption": metadata.get("qwen_caption", ""),
        "user_requests": parsed,
        "persona": metadata.get("persona", ""),
    }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input JSONL file."""
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

    # Limit samples if specified (shuffle first for diversity)
    if args.num_samples > 0:
        random.seed(RANDOM_SEED)
        random.shuffle(samples)
        samples = samples[:args.num_samples]
        print(f"Shuffled and selected {len(samples)} samples (seed={RANDOM_SEED})")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "user_requests.jsonl")
    persona_cache_file = os.path.join(args.output_dir, ".persona_cache.jsonl")

    # Check for existing progress using index-based tracking
    processed_indices = set()
    if os.path.exists(output_file) and args.resume:
        processed_indices = get_processed_indices(output_file)
        print(f"Resuming: found {len(processed_indices)} already processed samples")
    else:
        # Clear output files
        open(output_file, "w").close()
        if os.path.exists(persona_cache_file):
            os.remove(persona_cache_file)

    # Parse URLs and initialize processor
    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    num_servers = len(base_urls)
    workers_per_server = max(1, args.num_workers // num_servers)

    # Index samples
    indexed_samples = {idx: sample for idx, sample in enumerate(samples)}

    # Filter to only unprocessed samples
    pending_indices = [
        idx for idx in range(len(samples))
        if idx not in processed_indices
    ]
    print(f"Samples remaining to process: {len(pending_indices)}")

    if not pending_indices:
        print("All samples already processed!")
        return

    # =================================================================
    # Stage 1: Persona Selection
    # =================================================================
    print("\n" + "=" * 60)
    print("Stage 1: Persona Selection")
    print("=" * 60)

    # Check persona cache for resume
    cached_personas: Dict[int, str] = {}
    if os.path.exists(persona_cache_file) and args.resume:
        with open(persona_cache_file, 'r') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached_personas[rec['idx']] = rec['persona']
        print(f"Loaded {len(cached_personas)} cached personas")

    # Build persona queries for samples not in cache
    persona_queries = []
    for idx in pending_indices:
        if idx in cached_personas:
            continue
        query = build_persona_query(idx, indexed_samples[idx], prompts)
        if query is not None:
            persona_queries.append(query)

    print(f"Persona queries to process: {len(persona_queries)}")

    if persona_queries:
        processor = MassiveQueryProcessor(
            base_urls=base_urls,
            model=model,
            workers_per_server=workers_per_server,
            timeout=args.timeout,
            checkpoint_interval=10000,
        )

        def process_persona_result(result: Dict) -> Optional[Dict]:
            """Process persona selection result and save to cache."""
            response = result.get('response')
            persona = parse_persona_response(response)
            if persona is None:
                return None
            return {'idx': result['idx'], 'persona': persona}

        await processor.process_all(
            queries=persona_queries,
            output_file=persona_cache_file,
            process_fn=process_persona_result,
        )

    # Load all personas from cache
    if os.path.exists(persona_cache_file):
        with open(persona_cache_file, 'r') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached_personas[rec['idx']] = rec['persona']

    print(f"Total personas available: {len(cached_personas)}")

    # =================================================================
    # Stage 2: User Request Generation
    # =================================================================
    print("\n" + "=" * 60)
    print("Stage 2: User Request Generation")
    print("=" * 60)

    # Build request queries for samples with personas
    request_queries = []
    for idx in pending_indices:
        if idx not in cached_personas:
            continue
        persona = cached_personas[idx]
        query = build_request_query(idx, indexed_samples[idx], persona, prompts)
        request_queries.append(query)

    print(f"Request queries to process: {len(request_queries)}")

    if request_queries:
        processor = MassiveQueryProcessor(
            base_urls=base_urls,
            model=model,
            workers_per_server=workers_per_server,
            timeout=args.timeout,
            checkpoint_interval=10000,
        )

        await processor.process_all(
            queries=request_queries,
            output_file=output_file,
            process_fn=process_request_result,
        )

    # Final summary
    final_count = len(get_processed_indices(output_file))
    print(f"\nDone! Total success: {final_count}/{len(samples)}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate user requests from rich captions using vLLM."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with rich captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for user requests.",
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
