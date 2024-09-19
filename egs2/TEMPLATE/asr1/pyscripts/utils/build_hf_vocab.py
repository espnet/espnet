import argparse
import json

from transformers import AutoConfig, AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="Parse the BPE vocabulary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        required=True,
        help="model tag for huggingface model",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # (1) all vocabulary
    tokenizer = AutoTokenizer.from_pretrained(args.model_tag)
    token_and_ids = [(token, tid) for token, tid in tokenizer.get_vocab().items()]
    token_and_ids.sort(key=lambda x: x[1])

    token_list = []

    for idx in range(len(token_and_ids)):
        token, tid = token_and_ids[idx]
        assert tid == idx
        token_list.append(token)

    # (2) empty slots.
    vocab_size = AutoConfig.from_pretrained(args.model_tag).vocab_size
    for idx in range(len(token_and_ids), vocab_size):
        token_list.append(f"<unused_text_bpe_{idx}>")

    # (3) dump list
    print(json.dumps(token_list, indent=4))


if __name__ == "__main__":
    main()
