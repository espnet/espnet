import webdataset as wds
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    # currently 6 partitions (12/5/2024) but could change
    for i in range(1, 7):
        # keep trying until it succeeds
        while True:
            try:
                print(f"downloading anyspeech/ipapack_plus_{i}")
                snapshot_download(
                    repo_id=f"anyspeech/ipapack_plus_{i}",
                    repo_type="dataset",
                    local_dir="downloads",
                    local_dir_use_symlinks=False,
                    resume_download=False,
                    max_workers=4,
                )
                print(f"finished anyspeech/ipapack_plus_{i}")
                break
            except Exception as e:
                print(f"Retrying: {e}")
                continue
    print("All done")
