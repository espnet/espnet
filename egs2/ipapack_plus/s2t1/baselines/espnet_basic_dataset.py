# Read ESPnet style wav.scp and text files to return data instances
import torchaudio
from torch.utils.data import Dataset

class ESPnetBasicDataset(Dataset):
    def __init__(self, path, sampling_rate=16000):
        self.path = path
        self.sampling_rate = sampling_rate
        with open(f"{path}/wav.scp", "r") as f:
            self.wav_scp = {line.strip().split()[0]: line.strip().split()[1] for line in f.readlines()}
        with open(f"{path}/text.good", "r") as f:
            # text.good has phonemic sequences
            self.text = {line.strip().split()[0]: line.strip().split()[1] for line in f.readlines()}
        assert set(self.wav_scp.keys()) == set(self.text.keys()), "Mismatch between wav.scp and text keys"
        self.keys = list(self.wav_scp.keys())
    
    def __len__(self):
        return len(self.wav_scp)

    def __getitem__(self, idx):
        """Returns a dictionary with keys:
        'key': utterance ID
        'wav': waveform tensor (1, T)
        'transcription': phonemic transcription string
        'wavpath': path to the waveform file -- needed for some models (they read from file directly)
        """
        kidx=self.keys[idx]
        wav_path = self.wav_scp[kidx]
        transcription = self.text[kidx]
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return {"key": kidx, "wav": waveform, "transcription": transcription, 'wavpath': wav_path}


def get_inference_dataset(dataset, **kwargs):
    assert dataset in ['aishell', 'buckeye', 
                       'cv', 'doreco', 'epadb', 
                       'fleurs', 'fleurs_indv', 
                       'kazakh', 'l2arctic', 'librispeech', 
                       'mls_dutch', 'mls_french', 'mls_german', 
                       'mls_italian', 'mls_polish', 'mls_portuguese', 
                       'mls_spanish', 'southengland', 'speechoceannotth', 
                       'tamil', 'tusom2021', 'voxangeles'], f"Unknown dataset: {dataset}"
    return ESPnetBasicDataset(path=f"/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/dump/raw/test_{dataset}", **kwargs)


if __name__ == "__main__":
    dataset = get_inference_dataset("buckeye")
    for i in range(len(dataset)):
        item = dataset[i]
        print(item["key"], item["wav"].shape, item["transcription"])
        break
