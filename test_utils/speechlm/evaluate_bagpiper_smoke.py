#!/usr/bin/env python3
# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Check generated smoke audio, optionally with independent greedy CTC ASR.

The ASR comparison is a content diagnostic, not a Bagpiper benchmark score.
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def word_distance(reference, hypothesis):
    """Levenshtein distance over already normalized word sequences."""
    row = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        next_row = [i]
        for j, other in enumerate(hypothesis, 1):
            next_row.append(
                min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (word != other))
            )
        row = next_row
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()
    references = {
        row["id"]: row["transcript"] for row in json.loads(args.samples.read_text())
    }
    paths = sorted(args.results.rglob("*.wav"))
    if not paths or (
        args.expected_count is not None and len(paths) != args.expected_count
    ):
        raise ValueError(
            f"Expected {args.expected_count or 'nonzero'} WAVs, got {len(paths)}"
        )

    if args.asr:
        import torch
        import torchaudio

        torch.set_num_threads(8)
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        recognizer = bundle.get_model().eval()
        labels = bundle.get_labels()

    results, seen = [], set()
    for path in paths:
        example_id = path.name.split("_segment")[0]
        if example_id in seen or example_id not in references:
            raise ValueError(f"Duplicate or unknown example ID: {example_id}")
        seen.add(example_id)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Empty or nonfinite audio: {path}")
        rms = float(np.mean(audio * audio) ** 0.5)
        if rms <= 1e-5:
            raise ValueError(f"Silent audio: {path}")
        item = {
            "id": example_id,
            "path": str(path),
            "seconds": len(audio) / rate,
            "sample_rate": rate,
            "channels": audio.shape[1],
            "rms": rms,
            "peak": float(np.abs(audio).max()),
            "reference": references[example_id],
        }
        if args.asr:
            waveform = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
            if rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, rate, bundle.sample_rate
                )
            with torch.inference_mode():
                emissions, _ = recognizer(waveform)
            tokens = [
                index
                for index, _ in itertools.groupby(emissions[0].argmax(-1).tolist())
                if index != 0
            ]
            hypothesis = "".join(labels[i] for i in tokens).replace("|", " ").strip()
            reference_words = re.sub(
                r"[^A-Z ]", "", references[example_id].upper()
            ).split()
            hypothesis_words = re.sub(r"[^A-Z ]", "", hypothesis.upper()).split()
            item.update(
                hypothesis=hypothesis,
                word_errors=word_distance(reference_words, hypothesis_words),
                reference_words=len(reference_words),
            )
        results.append(item)
        print(json.dumps(item), flush=True)

    summary = {
        "samples": len(results),
        "audio_seconds": sum(item["seconds"] for item in results),
        "min_seconds": min(item["seconds"] for item in results),
        "max_seconds": max(item["seconds"] for item in results),
    }
    if args.asr:
        errors = sum(item["word_errors"] for item in results)
        words = sum(item["reference_words"] for item in results)
        summary.update(
            recognizer="torchaudio/WAV2VEC2_ASR_BASE_960H",
            word_errors=errors,
            reference_words=words,
            diagnostic_wer=errors / words if words else None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "samples": results}, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
