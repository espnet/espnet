import argparse

from transformers import AutoTokenizer, AutoConfig

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
    token_and_ids.sort(key=lambda x:x[1])


    for idx in range(len(token_and_ids)):
        token, tid = token_and_ids[idx]
        assert tid == idx
        print(token)

    # (2) empty slots.
    vocab_size = AutoConfig.from_pretrained(args.model_tag).vocab_size
    for idx in range(len(token_and_ids), vocab_size):
        print(f"<unused_text_bpe_{idx}>")

if __name__ == "__main__":
    main()