import os
from pathlib import Path

# Directory containing your .wav files

# VGGSOUND
# wav_dir = Path(
#     "/work/hdd/bbjs/shared/corpora/vggsound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/audio"
# )
# wav_scp_path = "/work/nvme/bbjs/sbharadwaj/icme_challenge/dump/raw/vggsound/wav.scp"

wav_dir = Path("/work/hdd/bbjs/shared/corpora/EARS")
wav_scp_path = "/work/nvme/bbjs/sbharadwaj/fullas2m/data/ears/wav.scp"

os.makedirs(os.path.dirname(wav_scp_path), exist_ok=True)

with open(wav_scp_path, "w") as f:
    for wav_file in sorted(wav_dir.glob("*/*.wav")):
        utt_id = f"{wav_file.parent.stem}-{wav_file.stem}"
        f.write(f"{utt_id} {wav_file.resolve()}\n")

print(f"Written wav.scp with {sum(1 for _ in open(wav_scp_path))} entries.")
