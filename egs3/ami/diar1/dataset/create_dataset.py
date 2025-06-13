from lhotse.recipes.ami import prepare_ami
import lhotse
import os
from pathlib import Path

def get_ami_manifets(config, mics=["ihm-mix", "mdm", "sdm"]):
    root = config.ami_root
    import pdb
    pdb.set_trace()
    outdir = Path("./data/ami")
    outdir.mkdir(parents=True, exist_ok=True)

    for c_mic in mics:
        prepare_ami(data_dir=root, annotations_dir=root, output_dir=outdir, mic=c_mic)
    # this will prepare manifests


def get_cuts(config):

    # load recordings here
    dset = "ami"
    split = "train"
    N_JOBS = 16
    #fbank_config = FbankConfig()
    #fbank_config.num_mel_bins = config.features.num_mel_bins
    #extractor = Fbank()
    #extractor.config_type= fbank_config

    for mic in ["ihm-mix", "mdm", "sdm"]: # note add later SDM
        c_rec = lhotse.load_manifest(os.path.join("./data/ami", f"{dset}-{mic}_recordings_{split}.jsonl.gz"))
        c_sup = lhotse.load_manifest(os.path.join("./data/ami", f"{dset}-{mic}_supervisions_{split}.jsonl.gz"))
        cutset = lhotse.CutSet.from_manifests(c_rec, c_sup)
        # chunk into X seconds
        print(f"Splitting cutset for {dset} {split} {mic} in chunks. Number of jobs: {N_JOBS}")
        cutset = cutset.cut_into_windows(config.features.chunk_size, config.features.hop, num_jobs=N_JOBS, keep_excessive_supervisions=True)
        cutset = cutset.filter(lambda x: x.duration >= config.features.chunk_size)
        #print(f"Computing features and saving them to disk.")
        #cutset.compute_and_store_features(extractor=extractor,
        #                            storage_path=f'./dump/feats/{dset}-{mic}-{split}', num_jobs=N_JOBS)
        cutset.to_file(f"./data/{dset}/{dset}-{mic}-{split}-cuts.jsonl.gz")

    split = "dev"
    for mic in ["ihm-mix", "sdm"]: # note add later SDM
        c_rec = lhotse.load_manifest(os.path.join("./data/ami", f"{dset}-{mic}_recordings_{split}.jsonl.gz"))
        c_sup = lhotse.load_manifest(os.path.join("./data/ami", f"{dset}-{mic}_supervisions_{split}.jsonl.gz"))
        cutset = lhotse.CutSet.from_manifests(c_rec, c_sup)
        # chunk into X seconds
        #print(f"Splitting cutset for {dset} {split} {mic} in chunks. Number of jobs: {N_JOBS}")
        #cutset = cutset.cut_into_windows(config.features.chunk_size, config.features.hop, num_jobs=N_JOBS, keep_excessive_supervisions=True)
        #print(f"Computing features and saving them to disk.")
        #cutset.compute_and_store_features(extractor=extractor,
        #                            storage_path=f'./dump/feats/{dset}-{mic}-{split}', num_jobs=N_JOBS)
        cutset.to_file(f"./data/{dset}/{dset}-{mic}-{split}-cuts.jsonl.gz")

    split = "test"
    for mic in ["ihm-mix", "sdm"]:  # note add later SDM
        c_rec = lhotse.load_manifest(os.path.join("./data/ami", f"{dset}-{mic}_recordings_{split}.jsonl.gz"))
        c_sup = lhotse.load_manifest(os.path.join("./data/ami", f"{dset}-{mic}_supervisions_{split}.jsonl.gz"))
        cutset = lhotse.CutSet.from_manifests(c_rec, c_sup)
        # chunk into X seconds
        #print(f"Splitting cutset for {dset} {split} {mic} in chunks. Number of jobs: {N_JOBS}")
        #cutset = cutset.cut_into_windows(config.features.chunk_size, config.features.hop, num_jobs=N_JOBS, keep_excessive_supervisions=True)
        # print(f"Computing features and saving them to disk.")
        # cutset.compute_and_store_features(extractor=extractor,
        #                            storage_path=f'./dump/feats/{dset}-{mic}-{split}', num_jobs=N_JOBS)
        cutset.to_file(f"./data/{dset}/{dset}-{mic}-{split}-cuts.jsonl.gz")





