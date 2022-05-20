import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data into kaldi format")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("src", type=str)
    parser.add_argument("tgt", type=str)
    args = parser.parse_args()

    if args.tgt == "fr":
        tgt_path = "fra"
    elif args.tgt == "en":
        tgt_path = "eng"
    root = os.path.join(args.data_dir, f"{args.src}-{tgt_path}")

    for dataset in ["valid", "train"]:
        id_checker = set()
        target_path = f"data/{dataset}.{args.src}-{args.tgt}"
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        wav_root = os.path.join(root, dataset, "wav")
        label_root = os.path.join(root, dataset, "txt")
        label_wav = open(os.path.join(label_root, f"{dataset}.yaml"), "r")
        label_src = open(os.path.join(label_root, f"{dataset}.{args.src}"), "r")
        label_tgt = open(os.path.join(label_root, f"{dataset}.{tgt_path}"), "r")

        utt_id = open(os.path.join(target_path, "utt_id"), "w")

        wavscp = open(os.path.join(target_path, "wav.scp"), "w")
        utt2spk = open(os.path.join(target_path, "utt2spk"), "w")
        spk2utt = open(os.path.join(target_path, "spk2utt"), "w")
        text_src = open(os.path.join(target_path, args.src + ".org"), "w")
        text_tgt = open(os.path.join(target_path, args.tgt + ".org"), "w")
        reco2dur = open(os.path.join(target_path, "reco2dur"), "w")

        while True:
            wav = label_wav.readline()
            src = label_src.readline()
            tgt = label_tgt.readline()
            if not wav or not src or not tgt:
                break

            wav = wav.strip()[3:-1].split(" ")
            wav_path = wav[-1]
            duration = wav[1][:-1]
            if "wav" in wav_path:
                wav_id = wav_path[:-4]
            else:
                wav_id = wav_path
                wav_path += ".wav"

            if wav_id in id_checker:
                continue

            wavscp.write(
                "{} sox -t wavpcm {} -c 1 -r 16000 -t wavpcm - |\n".format(
                    wav_id, os.path.join(wav_root, wav_path)
                )
            )
            utt2spk.write("{} {}\n".format(wav_id, wav_id))
            spk2utt.write("{} {}\n".format(wav_id, wav_id))
            text_src.write("{}\n".format(src.strip()))
            text_tgt.write("{}\n".format(tgt.strip()))
            reco2dur.write("{} {}\n".format(wav_id, duration))
            utt_id.write("{}\n".format(wav_id))
            id_checker.add(wav_id)

    wavscp.close()
    utt2spk.close()
    spk2utt.close()
    text_src.close()
    text_tgt.close()
    utt_id.close()
