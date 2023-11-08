import argparse
import random
import shutil
from pathlib import Path
from typing import Union

from espnet2.utils.types import str2bool


def read_text(text: Path):
    uttid2text = {}
    with text.open("r") as fp:
        for line in fp.readlines():
            line = line.strip().split()
            uttid2text[line[0]] = " ".join(line[1:])
    return uttid2text


# Convert discrete unit to CJK characters
# CJK Unified Ideographs (20,977 chars): \u4e00 - \u9fff
def unit2cjk(unit: Union[str, int]) -> str:
    return chr(int("4e00", 16) + int(unit))


def cjk2unit(char: str) -> str:
    return str(ord(char) - int("4e00", 16))


def prepare_textlm(
    root: Path, out_dir=Path("data"), generate_text_token="<generatetext>"
):
    print("textlm:", root / "text")
    uttid2text = read_text(root / "text")
    res = []

    for uttid, text in uttid2text.items():
        uttid = f"textlm_{uttid}"
        text = text.lower()

        res.append(f"{uttid} {generate_text_token} {text}")

    # write res
    print("Creating textlm: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_speechlm(
    root: Path,
    out_dir=Path("data"),
    generate_speech_token="<generatespeech>",
    use_cjk=True,
):
    res = []
    uttid2token = read_text(root / f"token")
    for uttid in uttid2token:
        token = uttid2token[uttid].split()
        if use_cjk:
            token = [unit2cjk(t) for t in token]
        uttid = f"unitlm_{uttid}"
        token = "".join(token)

        res.append(f"{uttid} {generate_speech_token} {token}")

    print("Creating speechlm: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_asr(
    root: Path,
    out_dir=Path("data"),
    start_speech_token="<startofspeech>",
    generate_text_token="<generatetext>",
    use_cjk=True,
):
    uttid2text = read_text(root / f"text")
    uttid2token = read_text(root / f"token")
    res = []
    for uttid, text in uttid2text.items():
        token = uttid2token[uttid].split()
        if use_cjk:
            token = [unit2cjk(t) for t in token]

        uttid = f"asr_{uttid}"
        text = text.lower()
        token = "".join(token)

        res.append(f"{uttid} {start_speech_token}{token}{generate_text_token} {text}")

    # write res
    print("Creating asr: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


def prepare_tts(
    root: Path,
    out_dir=Path("data"),
    start_text_token="<startoftext>",
    generate_speech_token="<generatespeech>",
    use_cjk=True,
):
    uttid2text = read_text(root / f"text")
    uttid2token = read_text(root / f"token")
    res = []
    for uttid, text in uttid2text.items():
        token = uttid2token[uttid].split()
        if use_cjk:
            token = [unit2cjk(t) for t in token]

        uttid = f"tts_{uttid}"
        text = text.lower()
        token = "".join(token)

        res.append(f"{uttid} {start_text_token} {text}{generate_speech_token}{token}")

    print("Creating tts: ", out_dir / "lm_text")
    with (out_dir / "lm_text").open("a") as fp:
        for line in res:
            fp.write(f"{line}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help=" ", default="dump")
    parser.add_argument(
        "--start_text_token",
        type=str,
        help="Token to denote start of text as condition.",
        default="<startoftext>",
    )
    parser.add_argument(
        "--generate_text_token",
        type=str,
        help="Token to denote generate text.",
        default="<generatetext>",
    )
    parser.add_argument(
        "--start_speech_token",
        type=str,
        help="Token to denote start of speech as condition.",
        default="<startofspeech>",
    )
    parser.add_argument(
        "--generate_speech_token",
        type=str,
        help="Token to denote generate speech.",
        default="<generatespeech>",
    )
    parser.add_argument(
        "--use_cjk",
        type=str2bool,
        help="Whether to map speech tokens into cjk. Needed for BPE training",
        default=True,
    )

    args = parser.parse_args()
    out_dir = Path(args.path)

    # create empty file
    with (out_dir / "lm_text").open("w") as fp:
        print("Opened file:", out_dir)

    # prepare textlm
    prepare_textlm(
        out_dir / "text/textlm",
        out_dir=out_dir,
        generate_text_token=args.generate_text_token,
    )

    # process speechlm
    prepare_speechlm(
        out_dir / "speech/speechlm",
        out_dir=out_dir,
        generate_speech_token=args.generate_speech_token,
        use_cjk=args.use_cjk,
    )

    # process asr
    prepare_asr(
        out_dir / "speech/asr",
        out_dir=out_dir,
        start_speech_token=args.start_speech_token,
        generate_text_token=args.generate_text_token,
        use_cjk=args.use_cjk,
    )

    # process tts
    prepare_tts(
        out_dir / "speech/tts",
        out_dir=out_dir,
        start_text_token=args.start_text_token,
        generate_speech_token=args.generate_speech_token,
        use_cjk=args.use_cjk,
    )

    with (Path("data") / "nlsyms.txt").open("w") as fp:
        fp.write(
            "{}\n{}\n{}\n{}\n".format(
                args.start_text_token,
                args.generate_text_token,
                args.start_speech_token,
                args.generate_speech_token,
            )
        )
