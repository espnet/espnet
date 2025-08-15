import argparse
import logging
import os


def parser():
    parser = argparse.ArgumentParser(description="Prepare OOD test data")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="Directory of the dump dir, " "like /path/to/espnet/egs2/lid/spk1/dump",
    )

    parser.add_argument(
        "--train_set",
        type=str,
        required=True,
        help="Name of the train set in dump dir, like train_fleurs_lang",
    )

    # test sets, a list of str
    parser.add_argument(
        "--test_sets",
        type=str,
        required=True,
        help="Names of the test sets in dump dir, "
        "e.g. 'dev_fleurs_lang dev_ml_superb2_lang' ",
    )

    return parser.parse_args()


def main(args):
    # load train spk2utt
    train_spk2utt = os.path.join(args.dump_dir, args.train_set, "spk2utt")

    train_langs = set()
    with open(train_spk2utt, "r") as f:
        for line in f:
            lang = line.split(" ", 1)[0]  # Split only once at the first space
            train_langs.add(lang)
    logging.info(f"Train set {args.train_set} languages: {train_langs}")
    test_sets = args.test_sets.strip().split()
    for test_set in test_sets:
        test_spk2utt = os.path.join(args.dump_dir, test_set, "spk2utt")
        test_utt2spk = os.path.join(args.dump_dir, test_set, "utt2spk")
        test_wavscp = os.path.join(args.dump_dir, test_set, "wav.scp")

        test_langs = set()
        with open(test_spk2utt, "r") as f:
            for line in f:
                lang = line.split(" ", 1)[0]
                test_langs.add(lang)

        logging.info(f"Test set {test_set} languages: {test_langs}")

        test_train_lang_cross = test_langs.intersection(train_langs)

        logging.info(
            f"Test set {test_set} languages in "
            f"train set {args.train_set}: {test_train_lang_cross}"
        )

        cross_utt2spk = []
        cross_wavscp = []
        cross_utts = set()
        with open(test_utt2spk, "r") as f:
            for line in f:
                utt, lang = line.strip().split()
                if lang in test_train_lang_cross:
                    cross_utt2spk.append(f"{utt} {lang}\n")
                    cross_utts.add(utt)
        with open(test_wavscp, "r") as f:
            for line in f:
                utt, path = line.strip().split()
                if utt in cross_utts:
                    cross_wavscp.append(f"{utt} {path}\n")

        # output dir
        cross_dir = os.path.join(args.dump_dir, f"{test_set}_cross_{args.train_set}")
        os.makedirs(cross_dir, exist_ok=True)

        # write files
        with open(os.path.join(cross_dir, "utt2spk"), "w") as f:
            f.writelines(sorted(cross_utt2spk))
        with open(os.path.join(cross_dir, "wav.scp"), "w") as f:
            f.writelines(sorted(cross_wavscp))


if __name__ == "__main__":
    args = parser()
    main(args)
