import argparse
from pathlib import Path


def prepare_asr_tts(input, output, prefix, sos):
    """Only keep the condition but remove the true target."""

    with open(input, "r") as fin, open(output, "w") as fout:
        for line in fin.readlines():
            if line.startswith(prefix):
                fout.write(line.split(sos)[0] + sos + "\n")


def prepare_lm(input, output, prefix):
    """Keep the entire sentence to compute perplexity."""

    with open(input, "r") as fin, open(output, "w") as fout:
        for line in fin.readlines():
            if line.startswith(prefix):
                fout.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help=" ", default="dump")
    parser.add_argument("-t", "--test_file", type=str, help=" ")
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
    args = parser.parse_args()
    out_dir = Path(args.path)

    test_file = args.test_file  # "dump/raw/test/text"

    prepare_asr_tts(test_file, out_dir / "text.asr", "asr_", args.generate_text_token)

    prepare_asr_tts(test_file, out_dir / "text.tts", "tts_", args.generate_speech_token)
