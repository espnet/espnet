import argparse
import csv
import gzip
import os

import fairseq_speechmatrix.audio_utils as au
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def get_alignment_doc_path(base_folder, lang1, lang2):
    """
    Checks if the directory exists for the given language pair in either order.
    Returns the path of the alignment document and the order of languages.
    """
    p = os.path.join(base_folder, f"aligned_speech/{lang1}-{lang2}/{lang1}-{lang2}.tsv.gz")
    if os.path.exists(p):
        return p, (lang1, lang2)

    p = os.path.join(base_folder, f"aligned_speech/{lang2}-{lang1}/{lang2}-{lang1}.tsv.gz")
    if os.path.exists(p):
        return p, (lang2, lang1)

    raise FileNotFoundError(
        "Alignment document not found for either order of languages."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str)  # ${SPEECH_MATRIX}
    parser.add_argument("--src_langs", nargs="+")  # list of source languages
    parser.add_argument("--tgt_langs", nargs="+")  # list of source languages
    parser.add_argument("--subset", type=str)
    parser.add_argument("--save_folder", type=str)  # path for storing the data
    parser.add_argument("--dump_folder", type=str)  # path for storing the data
    parser.add_argument("--europarl_folder", type=str)
    parser.add_argument("--fleurs_folder", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if not os.path.exists(args.dump_folder):
        os.makedirs(args.dump_folder)

    if args.subset == "train":
        # generate required files for each language pairs
        for src_lang in args.src_langs:
            for tgt_lang in args.tgt_langs:
                if src_lang == tgt_lang:
                    continue
                print(
                    "Creating {} alignment audios between {} and {}.".format(
                        args.subset, src_lang, tgt_lang
                    )
                )

                src_directory_path = os.path.join(
                    args.save_folder, f"{args.subset}_{src_lang}_{tgt_lang}"
                )
                os.makedirs(src_directory_path, exist_ok=True)

                src_wavscp = open(
                    os.path.join(src_directory_path, f"wav.scp.{src_lang}"),
                    "w",
                    encoding="utf-8",
                )
                tgt_wavscp = open(
                    os.path.join(src_directory_path, f"wav.scp.{tgt_lang}"),
                    "w",
                    encoding="utf-8",
                )
                src_text = open(
                    os.path.join(src_directory_path, f"text.{src_lang}"),
                    "w",
                    encoding="utf-8",
                )
                tgt_text = open(
                    os.path.join(src_directory_path, f"text.{tgt_lang}"),
                    "w",
                    encoding="utf-8",
                )
                utt2spk = open(
                    os.path.join(src_directory_path, f"utt2spk"), "w", encoding="utf-8"
                )

                alignment_doc_path, _ = get_alignment_doc_path(
                    args.src_folder, src_lang, tgt_lang
                )

                with gzip.open(
                    alignment_doc_path, "rt", encoding="utf-8"
                ) as alignment_doc:
                    tsv_reader = csv.reader(alignment_doc, delimiter="\t")
                    next(tsv_reader, None)  # Skip the header row

                    # go thru all lines in tsv
                    for row in tqdm(tsv_reader):
                        if len(row) == 3:
                            _, lang1_audio_zip, lang2_audio_zip = row
                            if (src_lang, tgt_lang) == get_alignment_doc_path(
                                args.src_folder, src_lang, tgt_lang
                            )[1]:
                                src_audio_zip = lang1_audio_zip
                                tgt_audio_zip = lang2_audio_zip
                            else:
                                src_audio_zip = lang2_audio_zip  # Switch src and tgt
                                tgt_audio_zip = lang1_audio_zip

                            # reproducing audio pairs
                            src_wf = au.get_features_or_waveform(
                                os.path.join(args.src_folder, "audios", src_audio_zip),
                                need_waveform=True,
                            )
                            tgt_wf = au.get_features_or_waveform(
                                os.path.join(args.src_folder, "audios", tgt_audio_zip),
                                need_waveform=True,
                            )

                            src_directory_path = os.path.join(
                                args.dump_folder, "wavs", src_lang
                            )
                            os.makedirs(src_directory_path, exist_ok=True)
                            src_seg_path = os.path.join(
                                src_directory_path, f"{src_audio_zip}.wav"
                            )

                            tgt_directory_path = os.path.join(
                                args.dump_folder, "wavs", tgt_lang
                            )
                            os.makedirs(tgt_directory_path, exist_ok=True)
                            tgt_seg_path = os.path.join(
                                tgt_directory_path, f"{tgt_audio_zip}.wav"
                            )

                            sf.write(src_seg_path, src_wf, 16000)
                            sf.write(tgt_seg_path, tgt_wf, 16000)

                            # using the name src_audio_zip directly as uttid
                            src_wavscp.write(
                                "{} {}\n".format(src_audio_zip, src_seg_path)
                            )
                            tgt_wavscp.write(
                                "{} {}\n".format(src_audio_zip, tgt_seg_path)
                            )

                            # no text, so set all to "0"
                            src_text.write("{} {}\n".format(src_audio_zip, "0"))
                            tgt_text.write("{} {}\n".format(src_audio_zip, "0"))

                            # no speaker id, so set all to the same as uttid
                            utt2spk.write(
                                "{} {}\n".format(src_audio_zip, src_audio_zip)
                            )

                src_wavscp.close()
                tgt_wavscp.close()
                src_text.close()
                tgt_text.close()
                utt2spk.close()

    elif args.subset == "dev":
        # generate required files for each language pairs
        for src_lang in args.src_langs:
            for tgt_lang in args.tgt_langs:
                if src_lang == tgt_lang:
                    continue
                print(
                    "Creating {} alignment audios between {} and {}.".format(
                        args.subset, src_lang, tgt_lang
                    )
                )

                src_directory_path = os.path.join(
                    args.save_folder, f"{args.subset}_{src_lang}_{tgt_lang}"
                )
                os.makedirs(src_directory_path, exist_ok=True)

                src_wavscp = open(
                    os.path.join(src_directory_path, f"wav.scp.{src_lang}"),
                    "a",
                    encoding="utf-8",
                )
                tgt_wavscp = open(
                    os.path.join(src_directory_path, f"wav.scp.{tgt_lang}"),
                    "a",
                    encoding="utf-8",
                )
                src_text = open(
                    os.path.join(src_directory_path, f"text.{src_lang}"),
                    "a",
                    encoding="utf-8",
                )
                tgt_text = open(
                    os.path.join(src_directory_path, f"text.{tgt_lang}"),
                    "a",
                    encoding="utf-8",
                )
                utt2spk = open(
                    os.path.join(src_directory_path, f"utt2spk"), "a", encoding="utf-8"
                )

                src_alignment_doc_path = os.path.join(
                    args.fleurs_folder,
                    f"aud_manifests/{src_lang}-{tgt_lang}/valid_{src_lang}-{tgt_lang}_{src_lang}.tsv",
                )
                tgt_alignment_doc_path = os.path.join(
                    args.fleurs_folder,
                    f"aud_manifests/{src_lang}-{tgt_lang}/valid_{src_lang}-{tgt_lang}_{tgt_lang}.tsv",
                )

                tgt_text_path = os.path.join(
                    args.fleurs_folder,
                    f"s2u_manifests/{src_lang}-{tgt_lang}/valid_fleurs.{tgt_lang}",
                )

                src_df = pd.read_csv(src_alignment_doc_path, sep="\t")
                tgt_df = pd.read_csv(tgt_alignment_doc_path, sep="\t")
                tgt_list = [line.strip() for line in open(tgt_text_path).readlines()]
                assert len(src_df) == len(tgt_list)
                assert len(tgt_df) == len(tgt_list)

                src_speech_dir = src_df.columns[0]
                tgt_speech_dir = tgt_df.columns[0]

                for row_count in tqdm(range(len(src_df))):

                    src_directory_path = os.path.join(
                        src_speech_dir, src_df.index[row_count]
                    )
                    tgt_directory_path = os.path.join(
                        tgt_speech_dir, tgt_df.index[row_count]
                    )

                    src_wavscp.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:5], src_directory_path
                        )
                    )
                    tgt_wavscp.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:5], tgt_directory_path
                        )
                    )

                    src_text.write(
                        "{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], "0")
                    )
                    tgt_text.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:5], tgt_list[row_count]
                        )
                    )

                    # no speaker id, so set all to the same as uttid
                    utt2spk.write(
                        "{}_{} {}_{}\n".format(
                            src_lang,
                            src_df.index[row_count][:5],
                            src_lang,
                            src_df.index[row_count][:5],
                        )
                    )

                src_wavscp.close()
                tgt_wavscp.close()
                src_text.close()
                tgt_text.close()
                utt2spk.close()

                print(
                    "{} audio alignment between {} and {} finished, with {} audio pairs.".format(
                        args.subset, src_lang, tgt_lang, row_count
                    )
                )

    # test set
    else:
        # generate required files for each language pairs
        for src_lang in args.src_langs:
            for tgt_lang in args.tgt_langs:
                if src_lang == tgt_lang:
                    continue
                print(
                    "FLEURS: Creating {} alignment audios between {} and {}.".format(
                        args.subset, src_lang, tgt_lang
                    )
                )

                src_directory_path = os.path.join(
                    args.save_folder, f"{args.subset}_fleurs_{src_lang}_{tgt_lang}"
                )
                os.makedirs(src_directory_path, exist_ok=True)

                src_wavscp = open(
                    os.path.join(src_directory_path, f"wav.scp.{src_lang}"),
                    "a",
                    encoding="utf-8",
                )
                tgt_wavscp = open(
                    os.path.join(src_directory_path, f"wav.scp.{tgt_lang}"),
                    "a",
                    encoding="utf-8",
                )
                src_text = open(
                    os.path.join(src_directory_path, f"text.{src_lang}"),
                    "a",
                    encoding="utf-8",
                )
                tgt_text = open(
                    os.path.join(src_directory_path, f"text.{tgt_lang}"),
                    "a",
                    encoding="utf-8",
                )
                utt2spk = open(
                    os.path.join(src_directory_path, f"utt2spk"), "a", encoding="utf-8"
                )

                src_alignment_doc_path = os.path.join(
                    args.fleurs_folder,
                    f"aud_manifests/{src_lang}-{tgt_lang}/test_{src_lang}-{tgt_lang}_{src_lang}.tsv",
                )
                tgt_alignment_doc_path = os.path.join(
                    args.fleurs_folder,
                    f"aud_manifests/{src_lang}-{tgt_lang}/test_{src_lang}-{tgt_lang}_{tgt_lang}.tsv",
                )

                tgt_text_path = os.path.join(
                    args.fleurs_folder,
                    f"s2u_manifests/{src_lang}-{tgt_lang}/test_fleurs.{tgt_lang}",
                )

                src_df = pd.read_csv(src_alignment_doc_path, sep="\t")
                tgt_df = pd.read_csv(tgt_alignment_doc_path, sep="\t")
                tgt_list = [line.strip() for line in open(tgt_text_path).readlines()]
                assert len(src_df) == len(tgt_list)
                assert len(tgt_df) == len(tgt_list)

                src_speech_dir = src_df.columns[0]
                tgt_speech_dir = tgt_df.columns[0]

                for row_count in tqdm(range(len(src_df))):

                    src_directory_path = os.path.join(
                        src_speech_dir, src_df.index[row_count]
                    )
                    tgt_directory_path = os.path.join(
                        tgt_speech_dir, tgt_df.index[row_count]
                    )

                    src_wavscp.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:5], src_directory_path
                        )
                    )
                    tgt_wavscp.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:5], tgt_directory_path
                        )
                    )

                    src_text.write(
                        "{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], "0")
                    )
                    tgt_text.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:5], tgt_list[row_count]
                        )
                    )

                    # no speaker id, so set all to the same as uttid
                    utt2spk.write(
                        "{}_{} {}_{}\n".format(
                            src_lang,
                            src_df.index[row_count][:5],
                            src_lang,
                            src_df.index[row_count][:5],
                        )
                    )

                src_wavscp.close()
                tgt_wavscp.close()
                src_text.close()
                tgt_text.close()
                utt2spk.close()

                print(
                    "FLEURS: {} audio alignment between {} and {} finished, with {} audio pairs.".format(
                        args.subset, src_lang, tgt_lang, row_count
                    )
                )

                print(
                    "EPST: Creating {} alignment audios between {} and {}.".format(
                        args.subset, src_lang, tgt_lang
                    )
                )

                src_alignment_doc_path = os.path.join(
                    args.europarl_folder,
                    f"aud_manifests/{src_lang}-{tgt_lang}/test_epst_{src_lang}_{tgt_lang}.tsv",
                )
                tgt_text_path = os.path.join(
                    args.europarl_folder,
                    f"s2u_manifests/{src_lang}-{tgt_lang}/test_epst.{tgt_lang}",
                )

                # Some languages don't exist in europarl-st
                if not os.path.exists(src_alignment_doc_path):
                    print(
                        "EPST: Skipping because there is no language pair alignment between {} and {}.".format(
                            src_lang, tgt_lang
                        )
                    )
                    continue

                src_directory_path = os.path.join(
                    args.save_folder, f"{args.subset}_epst_{src_lang}_{tgt_lang}"
                )
                os.makedirs(src_directory_path, exist_ok=True)

                src_wavscp = open(
                    os.path.join(src_directory_path, f"wav.scp.{src_lang}"),
                    "a",
                    encoding="utf-8",
                )
                tgt_text = open(
                    os.path.join(src_directory_path, f"text.{tgt_lang}"),
                    "a",
                    encoding="utf-8",
                )
                utt2spk = open(
                    os.path.join(src_directory_path, f"utt2spk"), "a", encoding="utf-8"
                )

                src_df = pd.read_csv(src_alignment_doc_path, sep="\t")
                tgt_list = [line.strip() for line in open(tgt_text_path).readlines()]
                assert len(src_df) == len(tgt_list)

                src_speech_dir = src_df.columns[0]

                for row_count in tqdm(range(len(src_df))):
                    src_directory_path = os.path.join(
                        src_speech_dir, src_df.index[row_count]
                    )

                    # remove the .wav for uttid
                    src_wavscp.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:4], src_directory_path
                        )
                    )

                    tgt_text.write(
                        "{}_{} {}\n".format(
                            src_lang, src_df.index[row_count][:4], tgt_list[row_count]
                        )
                    )

                    # no speaker id, so set all to the same as uttid
                    utt2spk.write(
                        "{}_{} {}_{}\n".format(
                            src_lang,
                            src_df.index[row_count][:4],
                            src_lang,
                            src_df.index[row_count][:4],
                        )
                    )

                src_wavscp.close()
                tgt_text.close()
                utt2spk.close()

                print(
                    "EPST: {} audio alignment between {} and {} finished, with {} audio pairs.".format(
                        args.subset, src_lang, tgt_lang, row_count
                    )
                )
