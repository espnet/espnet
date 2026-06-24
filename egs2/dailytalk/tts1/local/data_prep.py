#!/usr/bin/env python3

import argparse
import re
import shlex
from pathlib import Path

PATTERN = re.compile(r"^(\d+)_([01])_d(\d+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dailytalk_root", type=Path)
    parser.add_argument("dialogue_list", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    dialogues = [line.strip() for line in args.dialogue_list.read_text().splitlines() if line.strip()]
    entries = []
    for dialogue in dialogues:
        dialogue_dir = args.dailytalk_root / "data" / dialogue
        for wav in dialogue_dir.glob("*.wav"):
            match = PATTERN.fullmatch(wav.stem)
            if match is None:
                raise RuntimeError(f"Unexpected DailyTalk filename: {wav}")
            if match.group(3) != dialogue:
                raise RuntimeError(f"Dialogue ID mismatch: {wav}")

            transcript_file = wav.with_suffix(".txt")
            if not transcript_file.is_file():
                raise RuntimeError(f"Missing transcript for {wav}")
            transcript = " ".join(transcript_file.read_text(encoding="utf-8").split())
            if not transcript:
                raise RuntimeError(f"Empty transcript: {transcript_file}")

            turn = int(match.group(1))
            speaker = f"spk{match.group(2)}"
            utt_id = f"{speaker}-d{int(dialogue):06d}-u{turn:03d}"
            wav_command = f"sox {shlex.quote(str(wav.resolve()))} -t wav - remix 1 |"
            entries.append((utt_id, wav_command, transcript, speaker))

    entries.sort(key=lambda item: item[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp, \
         (args.output_dir / "text").open("w", encoding="utf-8") as text, \
         (args.output_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk:
        for utt_id, wav, transcript, speaker in entries:
            wav_scp.write(f"{utt_id} {wav}\n")
            text.write(f"{utt_id} {transcript}\n")
            utt2spk.write(f"{utt_id} {speaker}\n")


if __name__ == "__main__":
    main()
