import argparse
import os
import re


def float2str(number, size=6):
    number = str(int(number * 1000))
    return (size - len(number)) * "0" + number


def load_rttm_text(path):
    # Read a RTTM file
    spk_index = 0
    data = {}
    spk_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = re.split(" +", line.rstrip())

            # RTTM format must have exactly 9 fields
            assert len(sps) == 10 and path
            label_type, utt_id, channel, start, duration, _, _, spk_id, _, _ = sps

            # Only support speaker label now
            assert label_type == "SPEAKER"

            if spk_id not in spk_dict.keys():
                spk_dict[spk_id] = spk_index
                spk_index += 1
            data[utt_id] = data.get(utt_id, []) + [
                (spk_id, float(start), float(start) + float(duration))
            ]

    return data, spk_dict


def process_metadata(metadata, target_dir, source_rttm, fs, num_spk):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    wavscp = open(os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8")
    spk2utt = open(os.path.join(target_dir, "spk2utt"), "w", encoding="utf-8")
    segments = open(os.path.join(target_dir, "segments"), "w", encoding="utf-8")
    rttm = open(os.path.join(target_dir, "rttm"), "w", encoding="utf-8")
    reco2dur = open(os.path.join(target_dir, "reco2dur"), "w", encoding="utf-8")

    spk2utt_cache = {}

    with open(metadata, "r", encoding="utf-8") as f:
        header = f.readline().split(",")
        if num_spk == 2:
            assert len(header) == 6
            for linenum, line in enumerate(f, 1):
                mix_id, mix_path, _, _, _, length = line.strip().split(",")
                # from 3536-8226-0026_1673-143397-0009
                # to 3536-8226-0026, 1673-143397-0009
                source1_id, source2_id = mix_id.split("_")
                spk1, spk2 = source1_id.split("-")[0], source2_id.split("-")[0]
                reco1, reco2 = source1_id[len(spk1) + 1 :], source2_id[len(spk2) + 1 :]
                wavscp.write("{} {}\n".format(mix_id, mix_path))
                spk1_segs, spk2_segs = source_rttm[reco1], source_rttm[reco2]

                for spk_id, start, end in spk1_segs:
                    assert spk_id == spk1
                    # making seg_id to have prefixes of spk_id to avoid sorting issue
                    seg_id = "{}-{}_{}_{}".format(
                        spk_id, mix_id, float2str(start), float2str(end)
                    )
                    segments.write("{} {} {} {}\n".format(seg_id, mix_id, start, end))
                    utt2spk.write("{} {}\n".format(seg_id, spk_id))
                    rttm.write(
                        "SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(
                            mix_id, start, end - start, spk_id
                        )
                    )
                    spk2utt_cache[spk_id] = spk2utt_cache.get(spk_id, []) + [mix_id]

                for spk_id, start, end in spk2_segs:
                    assert spk_id == spk2
                    # making seg_id to have prefixes of spk_id to avoid sorting issue
                    seg_id = "{}-{}_{}_{}".format(
                        spk_id, mix_id, float2str(start), float2str(end)
                    )
                    segments.write("{} {} {} {}\n".format(seg_id, mix_id, start, end))
                    utt2spk.write("{} {}\n".format(seg_id, spk_id))
                    rttm.write(
                        "SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(
                            mix_id, start, end - start, spk_id
                        )
                    )
                    spk2utt_cache[spk_id] = spk2utt_cache.get(spk_id, []) + [mix_id]

                reco2dur.write("{} {}\n".format(mix_id, float(length) / fs))

        elif num_spk == 3:
            assert len(header) == 7
            for linenum, line in enumerate(f, 1):
                mix_id, mix_path, _, _, _, _, length = line.strip().split(",")
                # from 6313-76958-0019_6295-244435-0031_2277-149897-0012
                # to 6313-76958-0019, 6295-244435-0031, 2277-149897-0012
                source1_id, source2_id, source3_id = mix_id.split("_")
                spk1, spk2, spk3 = (
                    source1_id.split("-")[0],
                    source2_id.split("-")[0],
                    source3_id.split("-")[0],
                )
                reco1, reco2, reco3 = (
                    source1_id[len(spk1) + 1 :],
                    source2_id[len(spk2) + 1 :],
                    source3_id[len(spk3) + 1 :],
                )
                wavscp.write("{} {}\n".format(mix_id, mix_path))
                spk1_segs, spk2_segs, spk3_segs = (
                    source_rttm[reco1],
                    source_rttm[reco2],
                    source_rttm[reco3],
                )

                for spk_id, start, end in spk1_segs:
                    assert spk_id == spk1
                    # making seg_id to have prefixes of spk_id to avoid sorting issue
                    seg_id = "{}-{}_{}_{}".format(
                        spk_id, mix_id, float2str(start), float2str(end)
                    )
                    segments.write("{} {} {} {}\n".format(seg_id, mix_id, start, end))
                    utt2spk.write("{} {}\n".format(seg_id, spk_id))
                    rttm.write(
                        "SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(
                            mix_id, start, end - start, spk_id
                        )
                    )
                    spk2utt_cache[spk_id] = spk2utt_cache.get(spk_id, []) + [mix_id]

                for spk_id, start, end in spk2_segs:
                    assert spk_id == spk2
                    # making seg_id to have prefixes of spk_id to avoid sorting issue
                    seg_id = "{}-{}_{}_{}".format(
                        spk_id, mix_id, float2str(start), float2str(end)
                    )
                    segments.write("{} {} {} {}\n".format(seg_id, mix_id, start, end))
                    utt2spk.write("{} {}\n".format(seg_id, spk_id))
                    rttm.write(
                        "SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(
                            mix_id, start, end - start, spk_id
                        )
                    )
                    spk2utt_cache[spk_id] = spk2utt_cache.get(spk_id, []) + [mix_id]

                for spk_id, start, end in spk3_segs:
                    assert spk_id == spk3
                    # making seg_id to have prefixes of spk_id to avoid sorting issue
                    seg_id = "{}-{}_{}_{}".format(
                        spk_id, mix_id, float2str(start), float2str(end)
                    )
                    segments.write("{} {} {} {}\n".format(seg_id, mix_id, start, end))
                    utt2spk.write("{} {}\n".format(seg_id, spk_id))
                    rttm.write(
                        "SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(
                            mix_id, start, end - start, spk_id
                        )
                    )
                    spk2utt_cache[spk_id] = spk2utt_cache.get(spk_id, []) + [mix_id]

                reco2dur.write("{} {}\n".format(mix_id, float(length) / fs))

    for spk_id in spk2utt_cache.keys():
        spk2utt.write("{} {}\n".format(spk_id, " ".join(spk2utt_cache[spk_id])))

    wavscp.close()
    utt2spk.close()
    spk2utt.close()
    segments.close()
    rttm.close()
    reco2dur.close()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--target_dir", type=str, required=True, help="Path to generate kaldi_style result"
)
parser.add_argument("--source_dir", type=str, default="Libri2Mix/wav16k/max/metadata")
parser.add_argument("--rttm_dir", type=str, default="metadata/LibriSpeech")
parser.add_argument("--fs", type=int, default=8000)
parser.add_argument("--num_spk", type=int, default=2)

args = parser.parse_args()

train_rttm, train_spk = load_rttm_text(
    os.path.join(args.rttm_dir, "train_clean_100.rttm")
)
dev_rttm, dev_spk = load_rttm_text(os.path.join(args.rttm_dir, "dev_clean.rttm"))
test_rttm, test_spk = load_rttm_text(os.path.join(args.rttm_dir, "test_clean.rttm"))

process_metadata(
    os.path.join(args.source_dir, "mixture_train-100_mix_both.csv"),
    os.path.join(args.target_dir, "train" + str(args.num_spk)),
    train_rttm,
    args.fs,
    args.num_spk,
)
process_metadata(
    os.path.join(args.source_dir, "mixture_dev_mix_both.csv"),
    os.path.join(args.target_dir, "dev" + str(args.num_spk)),
    dev_rttm,
    args.fs,
    args.num_spk,
)
process_metadata(
    os.path.join(args.source_dir, "mixture_test_mix_both.csv"),
    os.path.join(args.target_dir, "test" + str(args.num_spk)),
    test_rttm,
    args.fs,
    args.num_spk,
)

print("Successfully finish Kaldi-style preparation")
