import argparse
import glob
import os
from multiprocessing import Pool

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--njobs", type=int, default=4)
parser.add_argument("--myst_dir", type=str)
parser.add_argument("--multiprocessing", action="store_true")
args = parser.parse_args()


def flac2wav(filepath):
    assert filepath.endswith(".flac")
    outfilepath = filepath[:-5] + ".wav"
    cmd = f"ffmpeg -hide_banner -loglevel error -y -i {filepath} {outfilepath}"
    _ = os.system(cmd)
    if os.path.isfile(outfilepath):
        os.remove(filepath)
    return


# get file list
filepaths = glob.glob(os.path.join(args.myst_dir, "**/*.flac"), recursive=True)

# multiprocessing
if args.multiprocessing:
    with Pool(args.njobs) as pool, tqdm(total=len(filepaths)) as pbar:
        for _ in pool.imap(flac2wav, filepaths):
            pbar.update()
            pbar.refresh()
else:
    for filepath in tqdm(filepaths):
        flac2wav(filepath)
