import argparse
import os
import csv
import sys
import subprocess
import soundfile as sf
import pandas as pd

from espnet2.utils.types import str2bool

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
fairseq_path = os.path.join(current_script_dir, '..', 'fairseq')
fairseq_path = os.path.abspath(fairseq_path)

if fairseq_path not in sys.path:
    sys.path.append(fairseq_path)

import fairseq.data.audio.audio_utils as au


def get_alignment_doc_path(base_folder, lang1, lang2):
    """
    Checks if the directory exists for the given language pair in either order.
    Returns the path of the alignment document and the order of languages.
    """
    original_path = os.path.join(base_folder, f"aligned_speech/{lang1}-{lang2}/{lang1}-{lang2}.tsv")
    reversed_path = os.path.join(base_folder, f"aligned_speech/{lang2}-{lang1}/{lang2}-{lang1}.tsv")
    if os.path.exists(original_path):
        return original_path, (lang1, lang2)
    elif os.path.exists(reversed_path):
        return reversed_path, (lang2, lang1)
    else:
        raise FileNotFoundError("Alignment document not found for either order of languages.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str) # ${SPEECH_MATRIX}
    parser.add_argument("--src_langs", nargs='+') # list of source languages
    parser.add_argument("--tgt_langs", nargs='+') # list of source languages
    parser.add_argument("--subset", type=str) 
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--save_folder", type=str) # path for storing the data

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.subset == "train":
        # generate required files for each language pairs
        for src_lang in args.src_langs:
            for tgt_lang in args.tgt_langs:
                if src_lang == tgt_lang:
                    continue
                print("Creating {} alignment audios between {} and {}.".format(args.subset, src_lang, tgt_lang))

                src_directory_path = os.path.join(args.save_folder, f"{args.subset}_{src_lang}_{tgt_lang}")
                os.makedirs(src_directory_path, exist_ok=True)

                src_wavscp = open(os.path.join(src_directory_path, f"wav.scp.{src_lang}"), "w", encoding="utf-8")
                tgt_wavscp = open(os.path.join(src_directory_path, f"wav.scp.{tgt_lang}"), "w", encoding="utf-8")
                src_text = open(os.path.join(src_directory_path, f"text.{src_lang}"), "w", encoding="utf-8")
                tgt_text = open(os.path.join(src_directory_path, f"text.{tgt_lang}"), "w", encoding="utf-8")
                utt2spk = open(os.path.join(src_directory_path, f"utt2spk"), "w", encoding="utf-8")

                alignment_doc_path, (src_lang, tgt_lang) = get_alignment_doc_path(args.src_folder, src_lang, tgt_lang)

                with open(alignment_doc_path, "r", encoding="utf-8") as alignment_doc:
                    tsv_reader = csv.reader(alignment_doc, delimiter='\t')
                    next(tsv_reader, None)  # Skip the header row
        
                    # go thru all lines in tsv
                    row_count = 0
                    for row in tsv_reader:
                        row_count += 1
                        if len(row) == 3:
                            _, lang1_audio_zip, lang2_audio_zip = row
                            if (src_lang, tgt_lang) == get_alignment_doc_path(args.src_folder, src_lang, tgt_lang)[1]:
                                src_audio_zip = lang1_audio_zip
                                tgt_audio_zip = lang2_audio_zip
                            else:
                                src_audio_zip = lang2_audio_zip  # Switch src and tgt
                                tgt_audio_zip = lang1_audio_zip

                            # reproducing audio pairs
                            src_wf = au.get_features_or_waveform(
                                os.path.join(args.src_folder, "audios", src_lang, src_audio_zip),
                                need_waveform=True
                            )
                            tgt_wf = au.get_features_or_waveform(
                                os.path.join(args.src_folder, "audios", tgt_lang, tgt_audio_zip),
                                need_waveform=True
                            )

                            src_directory_path = os.path.join(args.src_folder, src_lang, args.subset)
                            os.makedirs(src_directory_path, exist_ok=True)
                            src_seg_path = os.path.join(src_directory_path, f"{src_audio_zip}.wav")
                            
                            tgt_directory_path = os.path.join(args.src_folder, tgt_lang, args.subset)
                            os.makedirs(tgt_directory_path, exist_ok=True)
                            tgt_seg_path = os.path.join(tgt_directory_path, f"{tgt_audio_zip}.wav")

                            sf.write(src_seg_path, src_wf, 16000)
                            sf.write(tgt_seg_path, tgt_wf, 16000)

                            src_wavscp.write("{} {}\n".format(src_audio_zip, src_seg_path))
                            tgt_wavscp.write("{} {}\n".format(src_audio_zip, tgt_seg_path))

                            # no text, so set all to "0"
                            src_text.write("{} {}\n".format(tgt_audio_zip, "0"))
                            tgt_text.write("{} {}\n".format(tgt_audio_zip, "0"))

                            # no speaker id, so set all to the same as uttid
                            utt2spk.write("{} {}\n".format(tgt_audio_zip, tgt_audio_zip))
                    
                        if row_count % 1000 == 0:
                            print(f"{row_count} language pairs between {src_lang} and {tgt_lang} are prepared.")

                src_wavscp.close()
                tgt_wavscp.close()
                src_text.close()
                tgt_text.close()
                utt2spk.close()

                print("Audio alignment between {} and {} finished, with {} audio pairs.".format(src_lang, tgt_lang, row_count))

    elif args.subset == "dev":
        # generate required files for each language pairs
        for src_lang in args.src_langs:
            for tgt_lang in args.tgt_langs:
                if src_lang == tgt_lang:
                    continue
                print("Creating {} alignment audios between {} and {}.".format(args.subset, src_lang, tgt_lang))

                src_directory_path = os.path.join(args.save_folder, f"{args.subset}_{src_lang}_{tgt_lang}")
                os.makedirs(src_directory_path, exist_ok=True)

                src_wavscp = open(os.path.join(src_directory_path, f"wav.scp.{src_lang}"), "a", encoding="utf-8")
                tgt_wavscp = open(os.path.join(src_directory_path, f"wav.scp.{tgt_lang}"), "a", encoding="utf-8")
                src_text = open(os.path.join(src_directory_path, f"text.{src_lang}"), "a", encoding="utf-8")
                tgt_text = open(os.path.join(src_directory_path, f"text.{tgt_lang}"), "a", encoding="utf-8")
                utt2spk = open(os.path.join(src_directory_path, f"utt2spk"), "a", encoding="utf-8")

                src_alignment_doc_path = os.path.join(args.src_folder, f"flores/aud_manifests/{src_lang}-{tgt_lang}/valid_{src_lang}-{tgt_lang}_{src_lang}.tsv")
                tgt_alignment_doc_path = os.path.join(args.src_folder, f"flores/aud_manifests/{src_lang}-{tgt_lang}/valid_{src_lang}-{tgt_lang}_{tgt_lang}.tsv")
                
                tgt_text_path = os.path.join(args.src_folder, f"flores/align/s2u_manifests/{src_lang}-{tgt_lang}/valid_fleurs.{tgt_lang}")

                src_df = pd.read_csv(src_alignment_doc_path, sep="\t")
                tgt_df = pd.read_csv(tgt_alignment_doc_path, sep="\t")
                tgt_list = [line.strip() for line in open(tgt_text_path).readlines()]
                assert len(src_df) == len(tgt_list)
                assert len(tgt_df) == len(tgt_list)

                src_speech_dir = src_df.columns[0]
                tgt_speech_dir = tgt_df.columns[0]

                for row_count in range(len(src_df)):

                    src_directory_path = os.path.join(src_speech_dir, src_df.index[row_count])   
                    tgt_directory_path = os.path.join(tgt_speech_dir, tgt_df.index[row_count])

                    src_wavscp.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], src_directory_path))
                    tgt_wavscp.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], tgt_directory_path))

                    src_text.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], "0"))
                    tgt_text.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], tgt_list[row_count]))

                    # no speaker id, so set all to the same as uttid
                    utt2spk.write("{}_{} {}_{}\n".format(src_lang, src_df.index[row_count][:5], src_lang, src_df.index[row_count][:5]))
        
                    if row_count+1 % 1000 == 0:
                        print(f"{row_count} language pairs between {src_lang} and {tgt_lang} are prepared.")

                src_wavscp.close()
                tgt_wavscp.close()
                src_text.close()
                tgt_text.close()
                utt2spk.close()

                print("{} audio alignment between {} and {} finished, with {} audio pairs.".format(args.subset, src_lang, tgt_lang, row_count))
    
    # test set
    else:
        if args.test_dataset == "flores":
            # generate required files for each language pairs
            for src_lang in args.src_langs:
                for tgt_lang in args.tgt_langs:
                    if src_lang == tgt_lang:
                        continue
                    print("Creating {} alignment audios between {} and {}.".format(args.subset, src_lang, tgt_lang))

                    src_directory_path = os.path.join(args.save_folder, f"{args.subset}_{src_lang}_{tgt_lang}")
                    os.makedirs(src_directory_path, exist_ok=True)

                    src_wavscp = open(os.path.join(src_directory_path, f"wav.scp.{src_lang}"), "a", encoding="utf-8")
                    tgt_wavscp = open(os.path.join(src_directory_path, f"wav.scp.{tgt_lang}"), "a", encoding="utf-8")
                    src_text = open(os.path.join(src_directory_path, f"text.{src_lang}"), "a", encoding="utf-8")
                    tgt_text = open(os.path.join(src_directory_path, f"text.{tgt_lang}"), "a", encoding="utf-8")
                    utt2spk = open(os.path.join(src_directory_path, f"utt2spk"), "a", encoding="utf-8")

                    src_alignment_doc_path = os.path.join(args.src_folder, f"flores/aud_manifests/{src_lang}-{tgt_lang}/test_{src_lang}-{tgt_lang}_{src_lang}.tsv")
                    tgt_alignment_doc_path = os.path.join(args.src_folder, f"flores/aud_manifests/{src_lang}-{tgt_lang}/test_{src_lang}-{tgt_lang}_{tgt_lang}.tsv")
                    
                    tgt_text_path = os.path.join(args.src_folder, f"flores/align/s2u_manifests/{src_lang}-{tgt_lang}/test_fleurs.{tgt_lang}")

                    src_df = pd.read_csv(src_alignment_doc_path, sep="\t")
                    tgt_df = pd.read_csv(tgt_alignment_doc_path, sep="\t")
                    tgt_list = [line.strip() for line in open(tgt_text_path).readlines()]
                    assert len(src_df) == len(tgt_list)
                    assert len(tgt_df) == len(tgt_list)

                    src_speech_dir = src_df.columns[0]
                    tgt_speech_dir = tgt_df.columns[0]


                    for row_count in range(len(src_df)):

                        src_directory_path = os.path.join(src_speech_dir, src_df.index[row_count])   
                        tgt_directory_path = os.path.join(tgt_speech_dir, tgt_df.index[row_count])

                        src_wavscp.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], src_directory_path))
                        tgt_wavscp.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], tgt_directory_path))

                        src_text.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], "0"))
                        tgt_text.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:5], tgt_list[row_count]))

                        # no speaker id, so set all to the same as uttid
                        utt2spk.write("{}_{} {}_{}\n".format(src_lang, src_df.index[row_count][:5], src_lang, src_df.index[row_count][:5]))
            
                        if row_count+1 % 1000 == 0:
                            print(f"{row_count} language pairs between {src_lang} and {tgt_lang} are prepared.")

                    src_wavscp.close()
                    tgt_wavscp.close()
                    src_text.close()
                    tgt_text.close()
                    utt2spk.close()

                    print("{} audio alignment between {} and {} finished, with {} audio pairs.".format(args.subset, src_lang, tgt_lang, row_count))
    
        elif args.test_dataset == "epst":
            # generate required files for each language pairs
            for src_lang in args.src_langs:
                for tgt_lang in args.tgt_langs:
                    if src_lang == tgt_lang:
                        continue
                    print("Creating {} alignment audios between {} and {}.".format(args.subset, src_lang, tgt_lang))

                    src_directory_path = os.path.join(args.save_folder, f"{args.subset}_{src_lang}_{tgt_lang}")
                    os.makedirs(src_directory_path, exist_ok=True)

                    src_wavscp = open(os.path.join(src_directory_path, f"wav.scp.{src_lang}"), "a", encoding="utf-8")
                    tgt_text = open(os.path.join(src_directory_path, f"text.{tgt_lang}"), "a", encoding="utf-8")
                    utt2spk = open(os.path.join(src_directory_path, f"utt2spk"), "a", encoding="utf-8")

                    src_alignment_doc_path = os.path.join(args.src_folder, f"epst/aud_manifests/{src_lang}-{tgt_lang}/test_epst_{src_lang}_{tgt_lang}.tsv")
                    
                    tgt_text_path = os.path.join(args.src_folder, f"epst/test/s2u_manifests/{src_lang}-{tgt_lang}/test_epst.{tgt_lang}")

                    src_df = pd.read_csv(src_alignment_doc_path, sep="\t")
                    tgt_list = [line.strip() for line in open(tgt_text_path).readlines()]
                    assert len(src_df) == len(tgt_list)

                    src_speech_dir = src_df.columns[0]

                    for row_count in range(len(src_df)):

                        src_directory_path = os.path.join(src_speech_dir, src_df.index[row_count])   

                        # remove the .wav for uttid
                        src_wavscp.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:4], src_directory_path))

                        tgt_text.write("{}_{} {}\n".format(src_lang, src_df.index[row_count][:4], tgt_list[row_count]))

                        # no speaker id, so set all to the same as uttid
                        utt2spk.write("{}_{} {}_{}\n".format(src_lang, src_df.index[row_count][:4], src_lang, src_df.index[row_count][:4]))
            
                        if row_count+1 % 1000 == 0:
                            print(f"{row_count} language pairs between {src_lang} and {tgt_lang} are prepared.")

                    src_wavscp.close()
                    tgt_text.close()
                    utt2spk.close()

                    print("{} audio alignment between {} and {} finished, with {} audio pairs.".format(args.subset, src_lang, tgt_lang, row_count))

        else:
            print("Wrong test dataset used, change test_data to flores or epst in local/data.sh")



    # Remove the fairseq_path from sys.path
    if fairseq_path in sys.path:
        sys.path.remove(fairseq_path)