from collections import defaultdict
from pathlib import Path


def prepare_multich_datadir(datadir_1ch, outdir):
    def parse_noisy_audio_uid(uid):
        # Noisy audio file name format:
        # Lab41-SRI-VOiCES-< room >-< distractor_noise >-
        #   sp< speaker_ID >-ch< chapter_ID >-seg< segment_ID >-
        #   mc< mic_ID >-< mic_type >-< mic_location >-dg< degree >.wav
        tup = uid.split("-")
        assert len(tup) == 12, tup
        sid = tup[5]
        assert sid.startswith("sp"), sid
        sid = sid[2:]
        tup[8] = r"mc.*"
        tup[10] = r".*"
        pattern = "-".join(tup)
        return sid, pattern

    # wav.scp and utt2spk
    utt2wavs = defaultdict(list)
    utt2spk = {}
    with (datadir_1ch / "wav.scp").open("r") as f:
        for line in f:
            uid, wavpath = line.strip().split(maxsplit=1)
            sid, pattern = parse_noisy_audio_uid(uid)
            # key = sid + "_" + pattern
            key = pattern
            utt2wavs[key].append(wavpath)
            utt2spk[key] = sid

    keys = sorted(utt2wavs.keys())
    with (outdir / "wav.scp").open("w") as f:
        for uid in keys:
            wavpaths = utt2wavs[uid]
            info = "sox -M " + " ".join(sorted(wavpaths))
            info += f" -c {len(wavpaths)} -t wav - |"
            f.write(f"{uid} {info}\n")
    with (outdir / "utt2spk").open("w") as f:
        for uid in keys:
            spk = utt2spk[uid]
            f.write(f"{uid} {spk}\n")

    # text
    utt2text = {}
    with (datadir_1ch / "text").open("r") as f:
        for line in f:
            uid, text = line.strip().split(maxsplit=1)
            sid, pattern = parse_noisy_audio_uid(uid)
            # key = sid + "_" + pattern
            key = pattern
            if key not in utt2text:
                utt2text[key] = text
            else:
                assert utt2text[key] == text, (line, utt2text[key], text)

    with (outdir / "text").open("w") as f:
        for uid in keys:
            text = utt2text[uid]
            f.write(f"{uid} {text}\n")

    # Source audio file name format:
    # Lab41-SRI-VOiCES-src-sp< speaker_ID >-ch< chapter_ID >-sg< segment_ID >.wav


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datadir_1ch", type=str, help="Path to the data directory for 1ch data"
    )
    parser.add_argument("--outdir", type=str, help="Path to the output data directory")
    args = parser.parse_args()

    datadir_1ch = Path(args.datadir_1ch)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prepare_multich_datadir(datadir_1ch, outdir)
