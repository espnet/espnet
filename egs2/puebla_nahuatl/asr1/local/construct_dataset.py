import os
from argparse import ArgumentParser


def LoadWavSpeakerInfo(info_file):
    """return dict of wav: spk_list"""

    info_file = open(info_file, "r", encoding="utf-8")
    raw_info = list(map((lambda x: x.split(",")), (info_file.read()).split("\n")))
    wav_spk_info = {}
    for mapping in raw_info:
        if len(mapping) < 3:
            continue
        [wav, spk1, spk2] = mapping
        wav_spk_info[wav] = [spk1]
        if spk2 != "":
            wav_spk_info[wav] += [spk2]
    return wav_spk_info


if __name__ == "__main__":
    parser = ArgumentParser(description="Process Raw data")
    parser.add_argument(
        "-w", dest="wav_path", type=str, help="wav path", default="remixed"
    )
    parser.add_argument(
        "-i",
        dest="speaker_info",
        type=str,
        help="speaker info file dir",
        default="local/speaker_wav_mapping_nahuatl_test.csv",
    )
    parser.add_argument(
        "-n", dest="new_data_dir", type=str, help="new data directory", default="packed"
    )

    args = parser.parse_args()
    if not os.path.isdir(args.new_data_dir):
        os.makedirs(args.new_data_dir)
    info = LoadWavSpeakerInfo(args.speaker_info)
    for root, dirs, files in os.walk(args.wav_path):
        for wav in files:
            if ".wav" not in wav:
                continue
            wav_id = wav[:12]
            if wav_id not in info:
                continue
            spks = info[wav_id]
            if "L" in wav:
                spk = spks[0]
            else:
                spk = spks[1]
            new_filename = wav[:-4] + "_%s.wav" % spk
            os.system(
                "cp %s %s"
                % (
                    os.path.join(root, wav),
                    os.path.join(args.new_data_dir, new_filename),
                )
            )
