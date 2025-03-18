import sys

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python download_binaural_librispeech.py <output_dir>")
        sys.exit(1)

    subset = sys.argv[1]
    output_dir = sys.argv[2]
    repo_id = "Holger1997/BinauralLibriSpeech"
    directory_name = "data/BinauralLibriSpeech"
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            f"{directory_name}/{subset}/*"
        ],  # Only download files in the specific folder
        local_dir=output_dir,
        repo_type="dataset",
    )
