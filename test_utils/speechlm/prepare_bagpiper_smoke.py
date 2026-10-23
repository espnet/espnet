#!/usr/bin/env python3
# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Prepare a deterministic LibriSpeech subset for the Bagpiper smoke guide.

This creates local manifests, not training data for a benchmark. After caption
inference, --captions builds paired audio/caption manifests for fine-tuning.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from lhotse import Recording, RecordingSet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--valid-count", type=int, default=32)
    parser.add_argument("--captions", type=Path)
    args = parser.parse_args()
    if args.train_count < 1 or args.valid_count < 1:
        parser.error("--train-count and --valid-count must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)

    transcripts = {}
    for path in sorted(args.librispeech.rglob("*.trans.txt")):
        for line in path.read_text().splitlines():
            key, text = line.split(maxsplit=1)
            transcripts[key] = text
    speakers = defaultdict(list)
    for path in sorted(args.librispeech.rglob("*.flac")):
        info = sf.info(path)
        if 2 <= info.duration <= 10:
            speakers[path.stem.split("-")[0]].append(path.resolve())
    selected = []
    if not speakers:
        raise ValueError(f"No 2–10 second FLAC clips found in {args.librispeech}")
    for index in range(max(map(len, speakers.values()))):
        for speaker in sorted(speakers, key=int):
            if index < len(speakers[speaker]):
                selected.append(speakers[speaker][index])
    count = args.train_count + args.valid_count
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} eligible clips for {count} requested")
    selected = selected[:count]
    records = [Recording.from_file(p, recording_id=p.stem) for p in selected]
    RecordingSet.from_recordings(records).to_file(audio_dir / "recordings.jsonl.gz")
    rows = [
        {
            "id": p.stem,
            "audio": str(p),
            "transcript": transcripts[p.stem],
            "seconds": r.duration,
            "split": "train" if i < args.train_count else "valid",
        }
        for i, (p, r) in enumerate(zip(selected, records))
    ]
    (output / "samples.json").write_text(json.dumps(rows, indent=2) + "\n")
    text_path = output / "transcripts.txt"
    text_path.write_text("".join(f"{r['id']} {r['transcript']}\n" for r in rows))
    if args.captions:
        captions = {}
        for path in sorted(args.captions.rglob("results.json")):
            for key, messages in json.loads(path.read_text()).items():
                values = [
                    msg[2] for msg in messages if msg[:2] == ["assistant", "text"]
                ]
                if (
                    len(values) != 1
                    or not isinstance(values[0], str)
                    or not values[0].strip()
                ):
                    raise ValueError(f"Expected one nonempty caption for {key}: {path}")
                if key in captions:
                    raise ValueError(f"Duplicate caption: {key}")
                captions[key] = values[0].replace("\n", " ").strip()
        missing = {r["id"] for r in rows} - captions.keys()
        if missing:
            raise ValueError(f"Missing captions for {sorted(missing)}")
        text_path = output / "captions.txt"
        text_path.write_text("".join(f"{r['id']} {captions[r['id']]}\n" for r in rows))
    entries = [
        {"name": "audio1", "path": str(audio_dir), "reader": "lhotse_audio"},
        {"name": "text1", "path": str(text_path), "reader": "text"},
    ]
    for split in ("all", "train", "valid"):
        ids = [r["id"] for r in rows if split == "all" or r["split"] == split]
        manifest = {"data_entry": entries, "samples": ids}
        (output / f"{split}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.captions:
        # Synthetic dialogue labels exercise the TTS recipe's text/audio turns.
        # They are smoke data, not replicas of the paper's instruction dataset.
        system = (
            "You are a helpful assistant that generates audio based on user requests. "
            "You can create various types of audio including sound effects, music, "
            "speech, ambient sounds, and any combination of these. When given a "
            "request, first think through what the user wants and how to create "
            "high-quality audio, then provide a detailed description of the audio you "
            "will generate."
        )
        dialogues = []
        for row in rows:
            messages = [
                ["system", "text", system],
                ["user", "text", f'A narrator reads: "{row["transcript"]}".'],
                [
                    "assistant",
                    "text",
                    "<think>\nUse one narrator with the voice and acoustic setting "
                    "described below.\n</think>\n\n" + captions[row["id"]],
                ],
                ["assistant", "audio", row["audio"]],
            ]
            dialogues.append({"example_id": row["id"], "messages": messages})
        dialogue_path = output / "dialogues.jsonl"
        dialogue_path.write_text("".join(json.dumps(d) + "\n" for d in dialogues))
        for split in ("all", "train", "valid"):
            ids = [r["id"] for r in rows if split == "all" or r["split"] == split]
            manifest = {
                "data_entry": [
                    {
                        "name": "dialogue",
                        "reader": "dialogue",
                        "path": str(dialogue_path),
                    }
                ],
                "samples": ids,
            }
            (output / f"tts_{split}.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    print(
        json.dumps(
            {
                "clips": len(rows),
                "seconds": sum(r.duration for r in records),
                "speakers": len({r["id"].split("-")[0] for r in rows}),
                "data": str(output),
                "captions": bool(args.captions),
            }
        )
    )


if __name__ == "__main__":
    main()
