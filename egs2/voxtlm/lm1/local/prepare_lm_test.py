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
    args = parser.parse_args()
    out_dir = Path(args.path)

    test_file = args.test_file  # "dump/raw/test/text"

    prepare_asr_tts(test_file, out_dir / "text.asr", "asr_", "<generatetext>")

    prepare_asr_tts(test_file, out_dir / "text.tts", "tts_", "<generatespeech>")

    prepare_lm(test_file, out_dir / "text.textlm", "textlm_")

    prepare_lm(test_file, out_dir / "text.speechlm", "speechlm_")
