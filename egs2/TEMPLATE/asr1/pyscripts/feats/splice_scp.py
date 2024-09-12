import argparse
import kaldiio
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssl_scp", 
        type=str, 
        help="scp of ssl tokens",
    )
    parser.add_argument(
        "--codec_scp", 
        type=str, 
        help="scp of codec tokens",
    )
    parser.add_argument(
        "--wspecifier", 
        type=str, 
        default=False,
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=1,
        help="tolerance to length mismatch"
    )
    parser.add_argument(
        "--ssl_vocab_size",
        type=int,
        help="size of ssl token vocab, a.k.a., bias of codec tokens"
    )
    parser.add_argument(
        "--codec_code_per_frame",
        type=int,
        default=8,
        help="number of codes per frame in codec"
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    ssl_reader = kaldiio.load_scp(args.ssl_scp)
    codec_reader = kaldiio.load_scp(args.codec_scp)
    writer = kaldiio.WriteHelper(args.wspecifier)

    for key in ssl_reader:
        ssl_value = ssl_reader[key].reshape(-1, 1)
        codec_value = codec_reader[key].reshape(-1, args.codec_code_per_frame)
        codec_value = codec_value + args.ssl_vocab_size

        if abs(len(ssl_value) - len(codec_value)) > args.tolerance:
            print(f"length mismatch: ssl: {len(ssl_value)} {len(codec_value)}")

        min_len = min(len(ssl_value), len(codec_value))
        ssl_value = ssl_value[:min_len]
        codec_value = codec_value[:min_len]

        value = np.concatenate([ssl_value, codec_value], axis=1)
        writer[key] = value.flatten().astype(np.int32)

if __name__ == "__main__":
    main()
