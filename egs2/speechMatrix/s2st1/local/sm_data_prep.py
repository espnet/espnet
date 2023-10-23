import argparse
import os
import csv

TGT_LANG = "en"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--dest", type=str)
    # parser.add_argument("--subset", type=str)
    parser.add_argument("--src_lang", type=str, default="es")

    args = parser.parse_args()

    # Open the output files
    tgt_wavscp = open(os.path.join(args.dest, "wav.scp.{}".format(TGT_LANG)), "w", encoding="utf-8")
    tgt_text = open(os.path.join(args.dest, "text.{}".format(TGT_LANG)), "w", encoding="utf-8")
    src_wavscp = open(os.path.join(args.dest, "wav.scp.{}".format(args.src_lang)), "w", encoding="utf-8")
    src_text = open(os.path.join(args.dest, "text.{}".format(args.src_lang)), "w", encoding="utf-8")

    # Create a dictionary for mapping src_audio to ID
    src_id_mapping = {} # es
    with open(os.path.join(args.datadir, "s2u_manifests/es-en/train_mined.tsv"), "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            src_id_mapping[row['src_audio']] = row['id']

    # Create a dictionary for mapping tgt_audio to ID
    tgt_id_mapping = {} # en
    with open(os.path.join(args.datadir, "s2u_manifests/en-es/train_mined.tsv"), "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            tgt_id_mapping[row['src_audio']] = row['id']

    # Read the manifest and populate the output files
    with open(os.path.join(args.datadir, "aligned_speech/en-es.tsv"), "r", encoding="utf-8") as manifest:
        for line in manifest:
            if line.startswith("score"):
                continue # Avoid the header line

            _, en_audio, es_audio = line.strip().split("\t")
            en_zip, en_offset, en_duration = en_audio.split(':')
            es_zip, es_offset, es_duration = es_audio.split(':')

            # Use the ID mapping to get the unique utterance ID
            utt_id_en = tgt_id_mapping.get(en_audio, None)
            utt_id_es = src_id_mapping.get(es_audio, None)
            
            if not utt_id_en or not utt_id_es:
                print(f"Warning: ID for {en_audio} or {es_audio} not found in mapping. Skipping...")
                continue

            # Write to wav.scp
            tgt_wavscp.write(
                "{} ffmpeg -i {} -f wav -ss {} -t {} -ar 16000 -ab 16 -ac 1 - |\n".format(
                    utt_id_en, os.path.join(args.datadir, en_zip), en_offset, en_duration
                )
            )
            src_wavscp.write(
                "{} ffmpeg -i {} -f wav -ss {} -t {} -ar 16000 -ab 16 -ac 1 - |\n".format(
                    utt_id_es, os.path.join(args.datadir, es_zip), es_offset, es_duration
                )
            )

    # read src/tgt text and populate output files
    with open(os.path.join(args.datadir, "s2u_manifests/es-en/source_unit/train_mined.tsv"), "r", encoding="utf-8") as src_man:
        for line in src_man:
            if line.startswith("id"):
                continue  # avoid header

            src_id, src_txt = line.strip().split("\t")

            src_text.write(
                "{} {}\n".format(src_id, src_txt)
            )

    with open(os.path.join(args.datadir, "s2u_manifests/en-es/source_unit/train_mined.tsv"), "r", encoding="utf-8") as tgt_man:
        for line in tgt_man:
            if line.startswith("id"):
                continue  # avoid header

            tgt_id, tgt_txt = line.strip().split("\t")

            tgt_text.write(
                "{} {}\n".format(tgt_id, tgt_txt)
            )

    # Close the output files
    tgt_wavscp.close()
    tgt_text.close()
    src_wavscp.close()
    src_text.close()
