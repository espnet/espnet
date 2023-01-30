import argparse
import os
import string

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
    "mexico-el",
    "voxpopuli",
]

SINGLE_LANG = ["eng1", "eng2", "eng3", "fra1", "fra2", "deu", "rus", "swa", "swe", "jpn", "cmn", "sat", "xty"]
LANG_TO_SELECTED_DATASET = {
    "eng1":"mls",
    "eng2":"nchlt",
    "eng3":"voxpopuli",
    "fra1":"voxforge",
    "fra2":"voxpopuli",
    "deu":"swc",
    "rus":"M-AILABS",
    "swa":"ALFFA",
    "swe": "NST",
    "jpn":"commonvoice",
    "cmn":"fleurs",
    "sat":"commonvoice",
    "xty":"mexico-el"
}


def process_text(text):
    return text.translate(str.maketrans("", "", string.punctuation)).upper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="eng")
    parser.add_argument("--duration", type=str, default="10min")
    parser.add_argument("--source", type=str, default="downloads")

    args = parser.parse_args()
    assert args.duration in ["10min", "1h"], "we only support 10min or 1h setting"
    assert args.lang in SINGLE_LANG, "the language {} is not in our recommend set".format(args.lang)

    langs_info = {}

    # process train
    train_wavscp = open(
        os.path.join("data", "train_{}_{}".format(args.duration, args.lang), "wav.scp"),
        "w",
        encoding="utf-8",
    )
    train_text = open(
        os.path.join("data", "train_{}_{}".format(args.duration, args.lang), "text"),
        "w",
        encoding="utf-8",
    )
    train_utt2spk = open(
        os.path.join("data", "train_{}_{}".format(args.duration, args.lang), "utt2spk"),
        "w",
        encoding="utf-8",
    )
    # process dev
    dev_wavscp = open(
        os.path.join("data", "dev_{}_{}".format(args.duration, args.lang), "wav.scp"),
        "w",
        encoding="utf-8",
    )
    dev_text = open(
        os.path.join("data", "dev_{}_{}".format(args.duration, args.lang), "text"),
        "w",
        encoding="utf-8",
    )
    dev_utt2spk = open(
        os.path.join("data", "dev_{}_{}".format(args.duration, args.lang), "utt2spk"),
        "w",
        encoding="utf-8",
    )
    # process test
    test_wavscp = open(
        os.path.join("data", "test_{}_{}".format(args.duration, args.lang), "wav.scp"),
        "w",
        encoding="utf-8",
    )
    test_text = open(
        os.path.join("data", "test_{}_{}".format(args.duration, args.lang), "text"),
        "w",
        encoding="utf-8",
    )
    test_utt2spk = open(
        os.path.join("data", "test_{}_{}".format(args.duration, args.lang), "utt2spk"),
        "w",
        encoding="utf-8",
    )

    # iterate through dataset
    for dataset in DATA:
        langs = [args.lang[:3]]
        for lang in langs:
            if not os.path.exists(os.path.join(args.source, dataset, lang)):
                continue
            if lang not in langs_info:
                langs_info[lang] = []
            langs_info[lang].append(dataset)

            # process train
            if dataset == LANG_TO_SELECTED_DATASET[args.lang]:
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
                        "{} sox {} -c 1 -t wavpcm -|\n".format(
                            utt_id,
                            os.path.join(
                                args.source, dataset, lang, "wav", "{}.wav".format(utt_id)
                            ),
                        )
                    )
                    train_text.write("{} {}\n".format(utt_id, process_text(text)))
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
                        "{} sox {} -c 1 -t wavpcm -|\n".format(
                            utt_id,
                            os.path.join(
                                args.source, dataset, lang, "wav", "{}.wav".format(utt_id)
                            ),
                        )
                    )
                    dev_text.write("{} {}\n".format(utt_id, process_text(text)))
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
                    "{} sox {} -c 1 -t wavpcm -|\n".format(
                        utt_id,
                        os.path.join(
                            args.source, dataset, lang, "wav", "{}.wav".format(utt_id)
                        ),
                    )
                )
                test_text.write("{} {}\n".format(utt_id, process_text(text)))
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
