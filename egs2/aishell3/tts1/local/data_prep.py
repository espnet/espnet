import argparse
import os

from espnet2.utils.types import str2bool

SPK_LABEL_LEN = 7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--external_g2p", type=str2bool, default=True)

    args = parser.parse_args()

    wav_dir = os.path.join(args.src, "wav")
    transcript = open(os.path.join(args.src, "content.txt"), "r", encoding="utf-8")

    wavscp = open(os.path.join(args.dest, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(args.dest, "utt2spk"), "w", encoding="utf-8")
    text = open(os.path.join(args.dest, "text"), "w", encoding="utf-8")

    while True:
        utt_info = transcript.readline()
        if not utt_info:
            break

        (wav_name, text_info) = utt_info.strip().split("\t")
        if args.external_g2p:
            text_info = "".join(text_info.split(" ")[::2])
        else:
            text_info = " ".join(text_info.split(" ")[1::2])

        spk_id = wav_name[:SPK_LABEL_LEN]
        utt_id = wav_name[:-4]

        wavscp.write("{} {}\n".format(utt_id, os.path.join(wav_dir, spk_id, wav_name)))
        utt2spk.write("{} {}\n".format(utt_id, spk_id))
        text.write("{} {}\n".format(utt_id, text_info))

    transcript.close()
    wavscp.close()
    utt2spk.close()
    text.close()
