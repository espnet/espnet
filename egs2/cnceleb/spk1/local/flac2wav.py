import argparse
import os
from multiprocessing import Pool

from tqdm import tqdm


def find_all_wav(dirname, extension="flac"):
    if dirname[-1] != os.sep:
        dirname += os.sep
    old_name = dirname.split("/")[-2]
    new_name = old_name.replace("flac", "wav")
    print("Find {}, waiting ...".format(extension))
    flac2wav = []
    for root, _, filenames in tqdm(os.walk(dirname, followlinks=True)):
        wav_files = [f for f in filenames if f.endswith(extension)]
        if len(wav_files) > 0:
            new_root = root.replace(old_name, new_name)
            if not os.path.exists(new_root):
                print(new_root)
                os.makedirs(new_root)
            for _wav in wav_files:
                old_path = os.path.join(root, _wav)
                new_path = os.path.join(new_root, _wav.replace(".flac", ".wav"))
                flac2wav.append([old_path, new_path])
    return flac2wav


def flac2wav_main(flac_list):
    flac_path, wav_path = flac_list[0], flac_list[1]
    print("convert {}".format(flac_path))
    if os.path.exists(wav_path):
        os.remove(wav_path)
    cmd = "sox -t flac {} -t wav -r 16k -b 16 {} channels 1".format(flac_path, wav_path)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data", help="dataset dir")
    parser.add_argument("--nj", type=int, default=8, help="number of processes")
    args = parser.parse_args()

    flac_list = find_all_wav(args.dataset_dir)
    with Pool(args.nj) as p:
        p.map(flac2wav_main, flac_list)
