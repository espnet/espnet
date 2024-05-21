import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="Input file with all training data"
    )
    parser.add_argument("-o", "--output", type=str, help="Output file for bpe training")
    parser.add_argument(
        "-n",
        "--num-utterances",
        type=int,
        default=700000,
        help="Number of utterances to be used for bpe traning",
    )

    args = parser.parse_args()

    i = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin.readlines():
            if line.startswith("asr") or line.startswith("tts"):
                fout.write(line)
                i = i + 1
                if i > args.num_utterances:
                    break

    print("Successfully created BPE training file")
