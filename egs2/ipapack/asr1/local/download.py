import webdataset as wds
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    for dataset in ["doreco", "fleurs", "mswc"]:
        snapshot_download(
            repo_id=f"anyspeech/{dataset}_ipa",
            repo_type="dataset",
            local_dir="downloads",
            local_dir_use_symlinks=False,
            resume_download=False,
            max_workers=4,
        )
