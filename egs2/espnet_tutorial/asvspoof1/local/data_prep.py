import argparse
import os

from espnet2.utils.types import str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--tgt", type=str)

    args = parser.parse_args()

    src_wav_path = os.path.join(args.src_folder, "espnet_{}".format(args.subset))
    if args.subset == "train":
        src_label_path = os.path.join(args.src_folder, "ASVspoof2019.LA.cm.espnet_{}.trn.txt".format(args.subset))
    else:
        src_label_path = os.path.join(args.src_folder, "ASVspoof2019.LA.cm.espnet_{}.trl.txt".format(args.subset))
    src_label = open(src_label_path, "r", encoding="utf-8")
    if not os.path.exists(args.tgt):
        os.makedirs(args.tgt)
    wavscp = open(os.path.join(args.tgt, "wav.scp"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt, "text"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(args.tgt, "utt2spk"), "w", encoding="utf-8")

    for line in src_label.readlines():
        line = line.strip().split(" ")
        if len(line) != 5:
            continue
        
        spk_id, utt_id, system, _, label = line
        wavscp.write("{}_{} {}/{}.flac\n".format(spk_id, utt_id, src_wav_path, utt_id))
        text.write("{}_{} {}\n".format(spk_id, utt_id, 1 if label == "spoof" else 0))
        utt2spk.write("{}_{} {}\n".format(spk_id, utt_id, spk_id))
    
    wavscp.close()
    text.close()
    utt2spk.close()