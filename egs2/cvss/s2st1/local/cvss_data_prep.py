import argparse
import os

from espnet2.utils.types import str2bool

TGT_LANG = "en"


def get_full_id(wavpath):
    full_id_info = {}
    wav_info = open(wavpath, "r", encoding="utf-8")
    for line in wav_info.readlines():
        full_utt_id = line.strip().split(maxsplit=1)[0]
        spk_id, utt_id = full_utt_id.split("-")
        full_id_info[utt_id] = full_utt_id
    return full_id_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--src_lang", type=str, default="de")

    args = parser.parse_args()

    full_id_info = get_full_id(os.path.join(args.dest, "wav.scp.{}".format(args.src_lang)))

    tgt_wavscp = open(os.path.join(args.dest, "wav.scp.{}".format(TGT_LANG)), "w", encoding="utf-8")
    tgt_text = open(os.path.join(args.dest, "text.{}".format(TGT_LANG)), "w", encoding="utf-8")

    manifest = open(os.path.join(args.datadir, "{}.tsv".format(args.subset)), "r", encoding="utf-8")
    for line in manifest.readlines():
        wav, text = line.strip().split("\t", maxsplit=1)
        wav = wav[:-4] # remove suffix .mp3
        if wav not in full_id_info.keys():
            print("{} not found, possibly due to data mismatch".format(wav))
        full_id = full_id_info[wav]
        tgt_wavscp.write("{} {}\n".format(full_id, os.path.join(args.datadir, args.subset, "{}.mp3.wav".format(wav))))
        tgt_text.write("{} {}\n".format(full_id, text))
    tgt_wavscp.close()
    tgt_text.close()
