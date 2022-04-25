from multiprocessing import Pool
import os
from pathlib import Path
import sys
import torchaudio
import tqdm

multi_path = sys.argv[1]
single_path = sys.argv[2]

data_list = ["tr_real", "tr_synthetic", "cv", "tt", "tt_qut"]


def m2s(pf):
    if ".wav" not in pf[2]:
        return
    mwav = os.path.join(pf[0], pf[2])
    maudio, sr = torchaudio.load(mwav)
    swav = os.path.join(pf[1], pf[2])
    saudio = maudio[0:1]
    torchaudio.save(swav, saudio, sr)
    return


for data in data_list:
    multi_folder = os.path.join(multi_path, data)
    single_folder = os.path.join(single_path, data)
    os.makedirs(os.path.join(single_folder, "mixture"), exist_ok=True)
    os.symlink(
        os.path.join(multi_folder, "s0_dry"), os.path.join(single_folder, "s0_dry")
    )
    os.symlink(
        os.path.join(multi_folder, "metadata.json"),
        os.path.join(single_folder, "metadata.json"),
    )
    for root, dirs, files in os.walk(os.path.join(multi_folder, "mixture")):
        pfiles = [
            (
                os.path.join(multi_folder, "mixture"),
                os.path.join(single_folder, "mixture"),
                f,
            )
            for f in files
        ]
        with Pool(processes=16) as p:
            with tqdm.tqdm(total=len(pfiles)) as pbar:
                for i, elem in enumerate(p.imap(m2s, pfiles)):
                    pbar.update()
