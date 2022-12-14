import json
from collections import defaultdict
from pathlib import Path


def get_spk2utt(paths, audio_format="flac"):
    spk2utt = defaultdict(list)
    for path in paths:
        for audio in Path(path).rglob("*.{}".format(audio_format)):
            readerID = audio.parent.parent.stem
            uid = audio.stem
            assert uid.split("-")[0] == readerID, audio
            spk2utt[readerID].append((uid, str(audio)))

    return spk2utt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_paths",
        type=str,
        nargs="+",
        help="Paths to Librispeech subsets",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="spk2utt_tse.json",
        help="Path to the output spk2utt json file",
    )
    parser.add_argument("--audio_format", type=str, default="flac")
    args = parser.parse_args()

    spk2utt = get_spk2utt(args.audio_paths, audio_format=args.audio_format)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(spk2utt, f)
