import argparse
import os

from espnet2.utils.types import str2bool

DATA = [
    "ALFFA",
    "LAD",
    "M-AILABS",
    "NST",
    "commonvoice",
    "fleurs",
    "googlei18n_asr",
    "googlei18n_tts",
    "mls",
    "nchlt",
    "swc",
    "voxforge",
]  # missing mexico and voxpopuli

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, default="train_10min")
    parser.add_argument("--train_dev", type=str, default="dev_10min")
    parser.add_argument("--test_set", type=str, default="test_10min")
    parser.add_argument("--duration", type=str, default="10min")
    parser.add_argument("--source", type=str, default="downloads")
    parser.add_argument("--lid", type=str2bool, default=False)

    args = parser.parse_args()
    assert args.duration in ["10min", "1h"], "we only "

    langs_info = {}

    # process train
    train_wavscp = open(
        os.path.join("data", args.train_set, "wav.scp"), "w", encoding="utf-8"
    )
    train_text = open(
        os.path.join("data", args.train_set, "text"), "w", encoding="utf-8"
    )
    train_utt2spk = open(
        os.path.join("data", args.train_set, "utt2spk"), "w", encoding="utf-8"
    )
    # process dev
    dev_wavscp = open(
        os.path.join("data", args.train_dev, "wav.scp"), "w", encoding="utf-8"
    )
    dev_text = open(os.path.join("data", args.train_dev, "text"), "w", encoding="utf-8")
    dev_utt2spk = open(
        os.path.join("data", args.train_dev, "utt2spk"), "w", encoding="utf-8"
    )
    # process test
    test_wavscp = open(
        os.path.join("data", args.test_set, "wav.scp"), "w", encoding="utf-8"
    )
    test_text = open(os.path.join("data", args.test_set, "text"), "w", encoding="utf-8")
    test_utt2spk = open(
        os.path.join("data", args.test_set, "utt2spk"), "w", encoding="utf-8"
    )

    # iterate through dataset
    for dataset in DATA:
        langs = os.listdir(os.path.join(args.source, dataset))
        for lang in langs:
            if lang not in langs_info:
                langs_info[lang] = []
            langs_info[lang].append(dataset)

            # process train
            train_transcript = open(
                os.path.join(
                    args.source,
                    dataset,
                    lang,
                    "transcript_{}_train.txt".format(args.duration),
                ),
                "r",
                encoding="utf-8",
            )
            for line in train_transcript.readlines():
                line = line.strip().split(maxsplit=2)
                utt_id, _, text = line
                train_wavscp.write(
                    "{} {}\n".format(
                        utt_id,
                        os.path.join(
                            args.source, dataset, lang, "wav", "{}.wav".format(utt_id)
                        ),
                    )
                )
                if args.lid:
                    train_text.write("{} [{}] {}\n".format(utt_id, lang, text))
                else:
                    train_text.write("{} {}\n".format(utt_id, text))
                train_utt2spk.write("{} {}\n".format(utt_id, utt_id))
            train_transcript.close()

            # process dev
            dev_transcript = open(
                os.path.join(args.source, dataset, lang, "transcript_10min_dev.txt"),
                "r",
                encoding="utf-8",
            )
            for line in dev_transcript.readlines():
                line = line.strip().split(maxsplit=2)
                utt_id, _, text = line
                dev_wavscp.write(
                    "{} {}\n".format(
                        utt_id,
                        os.path.join(
                            args.source, dataset, lang, "wav", "{}.wav".format(utt_id)
                        ),
                    )
                )
                if args.lid:
                    dev_text.write("{} [{}] {}\n".format(utt_id, lang, text))
                else:
                    dev_text.write("{} {}\n".format(utt_id, text))
                dev_utt2spk.write("{} {}\n".format(utt_id, utt_id))
            dev_transcript.close()

            # process test
            test_transcript = open(
                os.path.join(args.source, dataset, lang, "transcript_10min_test.txt"),
                "r",
                encoding="utf-8",
            )
            for line in test_transcript.readlines():
                line = line.strip().split(maxsplit=2)
                utt_id, _, text = line
                test_wavscp.write(
                    "{} {}\n".format(
                        utt_id,
                        os.path.join(
                            args.source, dataset, lang, "wav", "{}.wav".format(utt_id)
                        ),
                    )
                )
                if args.lid:
                    test_text.write("{} [{}] {}\n".format(utt_id, lang, text))
                else:
                    test_text.write("{} {}\n".format(utt_id, text))
                test_utt2spk.write("{} {}\n".format(utt_id, utt_id))
            test_transcript.close()

    train_wavscp.close()
    train_text.close()
    train_utt2spk.close()
    dev_wavscp.close()
    dev_text.close()
    dev_utt2spk.close()
    test_wavscp.close()
    test_text.close()
    test_utt2spk.close()
    print("{} languages processed, ".format(len(langs_info)))
    for lang in langs_info.keys():
        dataset = " ".join(langs_info[lang])
        print(f"{lang} {len(langs_info[lang])} {dataset}")
