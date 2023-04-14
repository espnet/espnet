import os
import glob
import sys
import random

class FeatureDataset():
    def __init__(self, feature_dir, metadata_dir, split, return_audio_path=False, split_version=0, low_quality_source=True):  # TODO: incoporate split_version into configuration. Change low quality to high quality source audios.
        if split == 'valid':
            split = 'validation'

        # TODO: this path has to be input from the config file
        # feature_dir = "/home/xingranc/MIR-Benchmark/data/MTG/hubert_features/HF_HuBERT_base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_ckpt_325_400000_feature_layer_all_reduce_mean"
        # metadata_dir = "/home/xingranc/MIR-Benchmark/src/mtg-jamendo-dataset"
        self.feature_dir = feature_dir
        self.metadata_dir = os.path.join(metadata_dir, f'autotagging_genre-{split}.tsv')
        
        self.split_version = split_version
        self.low_quality_source = low_quality_source
        self.metadata = open(self.metadata_dir, 'r').readlines()[1:]

        self.all_paths = [line.split('\t')[3] for line in self.metadata]
        self.all_tags = [line.split('\t')[5:] for line in self.metadata]
        self.all_speaker = [line.split('\t')[1] for line in self.metadata]
        self.all_track = [line.split('\t')[0] for line in self.metadata]
        self.all_album = [line.split('\t')[2] for line in self.metadata]

        assert len(self.all_paths) == len(self.all_tags) == len(self.metadata)
        # read class2id
        self.class2id = self.read_class2id(metadata_dir, split_version)
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path

    def __len__(self):
        return len(self.metadata)
    
    def read_class2id(self, metadata_dir, split_version):
        class2id = {}
        for split in ['train', 'validation', 'test']:
            data = open(os.path.join(metadata_dir, f'autotagging_genre-{split}.tsv'), "r").readlines()
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2id:
                        class2id[tag] = len(class2id)
        return class2id

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_prep.py [root]")
        sys.exit(1)
    root = sys.argv[1]

    all_audio_list = glob.glob(
        os.path.join(root, "*", "*.mp3")
    )
    audio_basename_set = set()
    for audio_path in all_audio_list:
        filename = os.path.basename(audio_path)
        audio_basename_set.add(filename.split('.')[0])

    train = FeatureDataset(root, os.path.join(root, "tsv"), 'train')
    valid = FeatureDataset(root, os.path.join(root, "tsv"), 'valid')
    test = FeatureDataset(root, os.path.join(root, "tsv"), 'test')

    with open(
        os.path.join('data', 'train', 'text'), 'w+'
    ) as fw_train_text, open(
        os.path.join('data', 'train', 'wav.scp'), 'w+'
    ) as fw_train_wavscp, open(
        os.path.join('data', 'train', 'utt2spk'), 'w+'
    ) as fw_train_utt2spk, open(
        os.path.join('data', 'test', 'text'), 'w+'
    ) as fw_test_text, open(
        os.path.join('data', 'test', 'wav.scp'), 'w+'
    ) as fw_test_wavscp, open(
        os.path.join('data', 'test', 'utt2spk'), 'w+'
    ) as fw_test_utt2spk, open(
        os.path.join('data', 'valid', 'text'), 'w+'
    ) as fw_valid_text, open(
        os.path.join('data', 'valid', 'wav.scp'), 'w+'
    ) as fw_valid_wavscp, open(
        os.path.join('data', 'valid', 'utt2spk'), 'w+'
    ) as fw_valid_utt2spk:
        for i in range(len(train.all_paths)):
            path = os.path.join(root, train.all_paths[i])
            low_quality_audio_file = path.replace('.mp3', '.low.mp3')
            tags = train.all_tags[i]
            track = train.all_track[i]
            speaker = train.all_speaker[i]
            album = train.all_album[i]
            uttid = "-".join([speaker, album, track])
            if os.path.exists(low_quality_audio_file):
                fw_train_text.write(f"{uttid} {tags}\n")
                fw_train_wavscp.write(f"{uttid} ffmpeg -i {low_quality_audio_file} -f wav -ar 16000 -ab 16 -ac 1 - |\n")
                fw_train_utt2spk.write(f"{uttid} {speaker}\n")
        
        for i in range(len(test.all_paths)):
            path = os.path.join(root, test.all_paths[i])
            low_quality_audio_file = path.replace('.mp3', '.low.mp3')
            tags = test.all_tags[i]
            track = test.all_track[i]
            speaker = test.all_speaker[i]
            album = test.all_album[i]
            uttid = "-".join([speaker, album, track])
            if os.path.exists(low_quality_audio_file):
                fw_test_text.write(f"{uttid} {tags}\n")
                fw_test_wavscp.write(f"{uttid} ffmpeg -i {low_quality_audio_file} -f wav -ar 16000 -ab 16 -ac 1 - |\n")
                fw_test_utt2spk.write(f"{uttid} {speaker}\n")

        for i in range(len(valid.all_paths)):
            path = os.path.join(root, valid.all_paths[i])
            uttid = valid.all_paths[i]
            low_quality_audio_file = path.replace('.mp3', '.low.mp3')
            tags = valid.all_tags[i]
            track = valid.all_track[i]
            speaker = valid.all_speaker[i]
            album = valid.all_album[i]
            uttid = "-".join([speaker, album, track])
            if os.path.exists(low_quality_audio_file):
                fw_valid_text.write(f"{uttid} {tags}\n")
                fw_valid_wavscp.write(f"{uttid} ffmpeg -i {low_quality_audio_file} -f wav -ar 16000 -ab 16 -ac 1 - |\n")
                fw_valid_utt2spk.write(f"{uttid} {speaker}\n")
