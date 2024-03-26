import argparse
import os
import csv
import sys
import subprocess
import soundfile as sf

from espnet2.utils.types import str2bool

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
fairseq_path = os.path.join(current_script_dir, '..', 'fairseq')
fairseq_path = os.path.abspath(fairseq_path)

if fairseq_path not in sys.path:
    sys.path.append(fairseq_path)

import fairseq.data.audio.audio_utils as au

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str) # ${SPEECH_MATRIX}
    parser.add_argument("--langs", nargs='+') # list of languages
    parser.add_argument("--subset", type=str) 
    parser.add_argument("--tgt", type=str) # path for storing the data

    args = parser.parse_args()

    if not os.path.exists(args.tgt):
        os.makedirs(args.tgt)


    if args.subset == "train":

        raw_audio_path = os.path.join(current_script_dir, '..', args.src_folder)

        # generate required files for each language pairs
        for src_lang in args.langs:
            for tgt_lang in args.langs:
                if src_lang >= tgt_lang:
                    continue
                print("Creating alignment audios between {} and {}.".format(src_lang, tgt_lang))

                src_src_wav_path = os.path.join(args.src_folder, "audios", src_lang)
                src_tgt_wav_path = os.path.join(args.src_folder, "audios", tgt_lang)
                src_alignment_path = os.path.join(
                    args.src_folder, "aligned_speech/{}-{}/{}-{}.tsv".format(src_lang, tgt_lang, src_lang, tgt_lang)
                )

                # append lines to the existing files or create new ones
                directory_path = os.path.join(args.tgt, args.subset)
                os.makedirs(directory_path, exist_ok=True)

                src_wavscp = open(os.path.join(directory_path, f"wav.scp.{src_lang}"), "a", encoding="utf-8")
                tgt_wavscp = open(os.path.join(directory_path, f"wav.scp.{tgt_lang}"), "a", encoding="utf-8")
                src_text = open(os.path.join(directory_path, f"text.{src_lang}"), "a", encoding="utf-8")
                tgt_text = open(os.path.join(directory_path, f"text.{tgt_lang}"), "a", encoding="utf-8")
                src_utt2spk = open(os.path.join(directory_path, f"utt2spk.{src_lang}"), "a", encoding="utf-8")
                tgt_utt2spk = open(os.path.join(directory_path, f"utt2spk.{tgt_lang}"), "a", encoding="utf-8")

                with open(src_alignment_path, "r", encoding="utf-8") as src_alignment:
                    tsv_reader = csv.reader(src_alignment, delimiter='\t')
                    next(tsv_reader, None)  # Skip the header row


                    # prepare the index of the wav to be stored
                    with open(os.path.join(directory_path, f"wav.scp.{src_lang}"), "r", encoding="utf-8") as file:
                        count_src = sum(1 for _ in file)
                    with open(os.path.join(directory_path, f"wav.scp.{tgt_lang}"), "r", encoding="utf-8") as file:
                        count_tgt = sum(1 for _ in file) 
        
                    # go thru all lines in tsv
                    row_count = 0
                    for row in tsv_reader:
                        row += 1
                        if len(row) == 3:
                            _, src_audio_zip, tgt_audio_zip = row # score src_audio   tgt_audio

                            count_src += 1
                            count_tgt += 1

                            # reproducing audio pairs
                            src_wf = au.get_features_or_waveform(
                                os.path.join(raw_audio_path, "audios", src_lang, src_audio_zip),
                                need_waveform=True
                            )
                            tgt_wf = au.get_features_or_waveform(
                                os.path.join(raw_audio_path, "audios", tgt_lang, tgt_audio_zip),
                                need_waveform=True
                            )

                            src_directory_path = os.path.join(raw_audio_path, src_lang, args.subset)
                            os.makedirs(src_directory_path, exist_ok=True)
                            src_seg_path = os.path.join(src_directory_path, f"{src_lang}_{count_src}.wav")
                            
                            tgt_directory_path = os.path.join(raw_audio_path, tgt_lang, args.subset)
                            os.makedirs(tgt_directory_path, exist_ok=True)
                            tgt_seg_path = os.path.join(tgt_directory_path, f"{tgt_lang}_{count_tgt}.wav")

                            sf.write(src_seg_path, src_wf, 16000)
                            sf.write(tgt_seg_path, tgt_wf, 16000)

                            src_wavscp.write("{}_{} {}\n".format(tgt_lang, count_tgt, src_seg_path))
                            tgt_wavscp.write("{}_{} {}\n".format(src_lang, count_src, tgt_seg_path))

                            # no text, so set all to " "
                            src_text.write("{}_{} {}\n".format(tgt_lang, count_tgt, " "))
                            tgt_text.write("{}_{} {}\n".format(src_lang, count_src, " "))

                            # no speaker id, so set all to 0
                            src_utt2spk.write("{}_{} {}\n".format(src_lang, count_src, 0))
                            tgt_utt2spk.write("{}_{} {}\n".format(tgt_lang, count_tgt, 0))
                    
                        if row_count % 1000 == 0:
                            print(f"{row_count} language pairs between {src_lang} and {tgt_lang} are prepared.")

                src_wavscp.close()
                tgt_wavscp.close()
                src_text.close()
                tgt_text.close()
                src_utt2spk.close()
                tgt_utt2spk.close()

                print("Audio alignment between {} and {} finished, with {} source and {} target audio.".format(src_lang, tgt_lang, count_src, count_tgt))


    # Remove the fairseq_path from sys.path
    if fairseq_path in sys.path:
        sys.path.remove(fairseq_path)