#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""LLM-as-judge quality validation for SFT training examples."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

SYSTEM_PROMPT = """You are a STRICT quality judge for text-to-audio training data. \
Your task is to critically evaluate training examples and identify any flaws or issues.

IMPORTANT: Be critical and look for problems. Most examples should score 3-4, \
not 5. A score of 5 means PERFECT with zero issues - this should be rare.

You must output ONLY valid JSON with the exact structure specified. \
No other text, explanation, or markdown formatting."""

USER_PROMPT_TEMPLATE = """Critically evaluate a text-to-audio training example. \
Your job is to FIND FLAWS and issues, not to praise the example.

## Scoring Guidelines (BE STRICT)
For each dimension, first identify any flaws/issues, then assign a score:

- Score 5 (Exceptional): ZERO flaws. Perfect. Could be used as a gold example. \
RARE - only ~10% of good examples deserve this.
- Score 4 (Good): Minor imperfections that don't significantly affect quality. \
Most good examples should be here.
- Score 3 (Acceptable): Noticeable issues but still usable for training.
- Score 2 (Poor): Significant issues that may harm training quality.
- Score 1 (Unacceptable): Critical flaws. Should not be used for training.

## Evaluation Dimensions

### Category 1: User Request Quality
- naturalness: Does it sound like a real human wrote it?
  * Flaws to look for: robotic phrasing, unnatural word order, overly formal \
language, sounds like a template, awkward sentence structure
- clarity: Is the intent clear and unambiguous?
  * Flaws to look for: vague goals, multiple interpretations possible, \
unclear what audio should contain
- specificity: Appropriate detail level?
  * Flaws to look for: too vague ("make some audio"), OR overly prescriptive \
with excessive technical details
- feasibility: Is this a reasonable audio generation request?
  * Flaws to look for: physically impossible sounds, unrealistic combinations, \
requests that can't be fulfilled by audio
- language_quality: Avoids technical audio jargon?
  * Flaws to look for: terms like "sidechain", "EQ", "reverb tail", \
"compression", "frequency response", "stereo imaging", "transient"

### Category 2: Request-Response Alignment
- intent_match: Does the caption address what the user wanted?
  * Flaws to look for: caption describes different audio than requested, \
misses the main point of request
- content_coverage: Are key elements from request in the caption?
  * Flaws to look for: important details from request missing in caption, \
key sounds/elements not mentioned
- no_contradictions: Does caption avoid contradicting the request?
  * Flaws to look for: request says X but caption says Y, conflicting details
- scope_match: Is response detail appropriate for request detail?
  * Flaws to look for: brief request gets overly detailed response, or \
detailed request gets sparse response
- transcription_match: If speech/lyrics exist, do they match EXACTLY?
  * Score 5 if NO transcription exists (non-speech audio)
  * Flaws to look for: different wording, missing quotes, altered text, \
transcription in only one place

### Category 3: Thinking Trace Quality
- reasoning_coherence: Does the reasoning flow logically from request to caption?
  * Flaws to look for: jumps in logic, non-sequiturs, reasoning steps that \
don't connect, conclusions not supported by prior reasoning
- step_completeness: Are all necessary reasoning steps present?
  * Flaws to look for: missing analysis of key request elements, skipped \
considerations, incomplete breakdown of what the audio should contain
- no_hallucination: Does the reasoning stay grounded in the request?
  * Flaws to look for: invented requirements not in request, assumed details \
without basis, fabricated constraints or preferences
- appropriate_depth: Is the reasoning appropriately detailed (not too shallow, not verbose)?
  * Flaws to look for: trivial one-line reasoning for complex requests, OR \
excessive rambling for simple requests, repetitive statements
- trace_caption_consistency: Does the thinking trace align with the final caption?
  * Flaws to look for: caption contains elements not reasoned about, reasoning \
mentions things missing from caption, contradictions between trace and output

### Category 4: Rich Caption Quality
- descriptiveness: Enough detail for audio generation?
  * Flaws to look for: too brief, missing key audio characteristics, \
lacks temporal/spatial information
- realism: Describes plausible audio?
  * Flaws to look for: physically impossible, contradictory sounds, \
unrealistic combinations
- coherence: Internally consistent?
  * Flaws to look for: contradicting elements within caption, \
inconsistent descriptions
- completeness: Covers necessary elements?
  * Flaws to look for: missing timing info, missing key sounds, \
incomplete atmosphere description

### Category 5: Overall Training Value
- learning_signal: Would training on this teach good behaviors?
  * Flaws to look for: trivial example, doesn't demonstrate useful patterns, \
could teach bad habits
- non_trivial: Requires actual understanding?
  * Flaws to look for: caption just copies request words, no reasoning needed, \
simple keyword matching would suffice
- persona_fit: Does persona make sense for this audio?
  * Flaws to look for: "voice-over artist" for non-speech, "musician" for \
pure sound effects, mismatched expertise

## Output Format
- If score < 5: include "flaws" field explaining the issues
- If score = 5: just include "score" (no flaws field needed)

Output JSON only:
{{"user_request_quality": {{"naturalness": {{"score": 5}}, \
"clarity": {{"flaws": "...", "score": 3}}, \
"specificity": {{"score": 4}}, \
"feasibility": {{"score": 5}}, \
"language_quality": {{"flaws": "...", "score": 2}}}}, \
"alignment": {{"intent_match": {{"score": 5}}, \
"content_coverage": {{"flaws": "...", "score": 4}}, \
"no_contradictions": {{"score": 5}}, \
"scope_match": {{"score": 4}}, \
"transcription_match": {{"score": 5}}}}, \
"thinking_trace_quality": {{"reasoning_coherence": {{"score": 4}}, \
"step_completeness": {{"flaws": "...", "score": 3}}, \
"no_hallucination": {{"score": 5}}, \
"appropriate_depth": {{"score": 4}}, \
"trace_caption_consistency": {{"score": 5}}}}, \
"caption_quality": {{"descriptiveness": {{"score": 4}}, \
"realism": {{"score": 5}}, \
"coherence": {{"score": 5}}, \
"completeness": {{"flaws": "...", "score": 3}}}}, \
"training_value": {{"learning_signal": {{"score": 4}}, \
"non_trivial": {{"flaws": "...", "score": 3}}, \
"persona_fit": {{"score": 5}}}}}}

## Training Example to Evaluate
**Persona:** {persona}
**Request Type:** {detail_level}

**User Request:**
{user_request}

**Thinking Trace (assistant's reasoning):**
{thinking_trace}

**Rich Caption (target audio description):**
{qwen_caption}"""

# Expected categories and dimensions for validation
EXPECTED_STRUCTURE = {
    "user_request_quality": [
        "naturalness", "clarity", "specificity", "feasibility", "language_quality"
    ],
    "alignment": [
        "intent_match", "content_coverage", "no_contradictions",
        "scope_match", "transcription_match"
    ],
    "thinking_trace_quality": [
        "reasoning_coherence", "step_completeness", "no_hallucination",
        "appropriate_depth", "trace_caption_consistency"
    ],
    "caption_quality": [
        "descriptiveness", "realism", "coherence", "completeness"
    ],
    "training_value": [
        "learning_signal", "non_trivial", "persona_fit"
    ],
}


def extract_text_components(dialogue: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract user request, thinking trace, rich caption, and persona from dialogue."""
    messages = dialogue.get("messages", [])
    if len(messages) < 2:
        return None

    user_request = None
    thinking_trace = None
    qwen_caption = None

    for msg in messages:
        role, modality, content = msg[0], msg[1], msg[2]
        if role == "user" and modality == "text":
            user_request = content
        elif role == "assistant" and modality == "text":
            # Extract thinking trace (between <think> and </think>)
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_start != -1 and think_end != -1:
                thinking_trace = content[think_start + len("<think>"):think_end].strip()
                qwen_caption = content[think_end + len("</think>"):].strip()
            else:
                # No think tag, use entire content as caption
                thinking_trace = ""
                qwen_caption = content.strip()

    if not all([user_request, qwen_caption]):
        return None

    # Extract persona from metadata
    metadata = dialogue.get("metadata", {})
    persona = metadata.get("persona", "unknown")

    return {
        "user_request": user_request,
        "thinking_trace": thinking_trace,
        "qwen_caption": qwen_caption,
        "persona": persona,
    }


def parse_judge_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from judge LLM."""
    if response is None:
        return None

    # Try direct parsing
    try:
        data = json.loads(response)
        if _validate_response_structure(data):
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from response
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            if _validate_response_structure(data):
                return data
    except json.JSONDecodeError:
        pass

    return None


def _validate_response_structure(data: Dict[str, Any]) -> bool:
    """Validate that response has expected structure.

    Only requires "score" field. "flaws" field is optional - we record it
    if present but don't require it even for scores < 5.
    """
    for category, dimensions in EXPECTED_STRUCTURE.items():
        if category not in data:
            return False
        for dim in dimensions:
            if dim not in data[category]:
                return False
            if not isinstance(data[category][dim], dict):
                return False
            if "score" not in data[category][dim]:
                return False
            if not isinstance(data[category][dim]["score"], (int, float)):
                return False
            # "flaws" field is optional - recorded if present but not required
    return True


def compute_overall_score(scores: Dict[str, Any]) -> Tuple[float, bool]:
    """Compute overall score and pass/fail status."""
    all_scores = []
    min_score = 5

    for category in EXPECTED_STRUCTURE.keys():
        if category in scores:
            for _, val in scores[category].items():
                if isinstance(val, dict) and "score" in val:
                    score = val["score"]
                    if isinstance(score, (int, float)):
                        all_scores.append(score)
                        min_score = min(min_score, score)

    if not all_scores:
        return 0.0, False

    avg_score = sum(all_scores) / len(all_scores)
    # Pass if: all scores >= 3 and average >= 3.5
    overall_pass = min_score >= 3 and avg_score >= 3.5

    return round(avg_score, 2), overall_pass


def build_judge_query(
    dialogue: Dict[str, Any],
    detail_level: str,
) -> Optional[Dict[str, Any]]:
    """Build a query for quality judgment."""
    components = extract_text_components(dialogue)
    if components is None:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                user_request=components["user_request"],
                thinking_trace=components["thinking_trace"],
                qwen_caption=components["qwen_caption"],
                persona=components["persona"],
                detail_level=detail_level,
            ),
        },
    ]

    return {
        'idx': dialogue.get("example_id", ""),
        'messages': messages,
        'temperature': 0.1,
        'max_tokens': 1536,
        'json_mode': True,
        'metadata': {
            'dialogue': dialogue,
            'detail_level': detail_level,
        },
    }


def process_judge_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a quality judgment result."""
    response = result.get('response')
    metadata = result.get('metadata', {})

    scores = parse_judge_response(response)
    if scores is None:
        return None

    overall_score, overall_pass = compute_overall_score(scores)

    return {
        "example_id": result['idx'],
        "detail_level": metadata.get('detail_level', ""),
        "scores": scores,
        "overall_score": overall_score,
        "overall_pass": overall_pass,
    }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input JSONL file with dialogues."""
    dialogues = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dialogues.append(json.loads(line))
    return dialogues


def compute_summary(results_file: str) -> Dict[str, Any]:
    """Compute summary statistics from judge results."""
    results = load_input_data(results_file)

    if not results:
        return {}

    total = len(results)
    passed = sum(1 for r in results if r.get("overall_pass", False))

    # Aggregate scores by category and dimension
    score_sums: Dict[str, Dict[str, float]] = {
        cat: {} for cat in EXPECTED_STRUCTURE.keys()
    }
    score_counts: Dict[str, Dict[str, int]] = {
        cat: {} for cat in EXPECTED_STRUCTURE.keys()
    }

    for r in results:
        scores = r.get("scores", {})
        for category in EXPECTED_STRUCTURE.keys():
            if category in scores and isinstance(scores[category], dict):
                for dim, val in scores[category].items():
                    if isinstance(val, dict) and "score" in val:
                        if dim not in score_sums[category]:
                            score_sums[category][dim] = 0
                            score_counts[category][dim] = 0
                        score_sums[category][dim] += val["score"]
                        score_counts[category][dim] += 1

    # Compute averages
    avg_scores: Dict[str, Dict[str, float]] = {}
    for category in EXPECTED_STRUCTURE.keys():
        avg_scores[category] = {}
        for dim in score_sums[category]:
            if score_counts[category][dim] > 0:
                avg_scores[category][dim] = round(
                    score_sums[category][dim] / score_counts[category][dim], 2
                )

    # Count failures by dimension (score < 3)
    failure_breakdown: Dict[str, int] = {}
    for r in results:
        scores = r.get("scores", {})
        for category in scores:
            if not isinstance(scores[category], dict):
                continue
            for dim, val in scores[category].items():
                if isinstance(val, dict) and "score" in val:
                    if val["score"] < 3:
                        key = f"{category}.{dim}"
                        failure_breakdown[key] = failure_breakdown.get(key, 0) + 1

    # Sort failure breakdown by count (descending)
    failure_breakdown = dict(
        sorted(failure_breakdown.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "total_samples": total,
        "passed_samples": passed,
        "pass_rate": round(passed / total, 4) if total > 0 else 0,
        "avg_scores": avg_scores,
        "failure_breakdown": failure_breakdown,
    }


async def main_async(args):
    """Main async function."""
    # Load input data
    print(f"Loading input data from {args.input_file}...")
    dialogues = load_input_data(args.input_file)
    print(f"Loaded {len(dialogues)} dialogues")

    # Limit samples if specified
    if args.num_samples > 0:
        dialogues = dialogues[:args.num_samples]
        print(f"Limited to {len(dialogues)} samples for processing")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"judge_{args.detail_level}.jsonl")

    # Check for existing progress using example_id-based tracking
    processed_ids = set()
    if os.path.exists(output_file) and args.resume:
        processed_ids = get_processed_indices(output_file, idx_key="example_id")
        print(f"Resuming: found {len(processed_ids)} already processed samples")
    else:
        open(output_file, "w").close()

    # Parse URLs and initialize processor
    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    num_servers = len(base_urls)
    workers_per_server = max(1, args.num_workers // num_servers)

    # Build queries for all pending dialogues
    queries = []
    for dialogue in dialogues:
        example_id = dialogue.get("example_id")
        if example_id in processed_ids:
            continue
        query = build_judge_query(dialogue, args.detail_level)
        if query is not None:
            queries.append(query)

    print(f"Queries to process: {len(queries)}")

    if not queries:
        print("All dialogues already processed!")
    else:
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
            process_fn=process_judge_result,
        )

    # Compute and save summary
    summary = compute_summary(output_file)
    summary_file = os.path.join(args.output_dir, f"summary_{args.detail_level}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    final_count = len(get_processed_indices(output_file, idx_key="example_id"))
    print(f"\nDone! Total success: {final_count}/{len(dialogues)}")
    print(f"Pass rate: {summary.get('pass_rate', 0):.1%}")
    print(f"Output saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge quality validation for SFT training examples."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with dialogues.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for judge results.",
    )
    parser.add_argument(
        "--detail_level",
        type=str,
        required=True,
        choices=["realistic", "imaginary"],
        help="Detail level being judged.",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL (use ':' to separate multiple URLs).",
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
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress.",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
