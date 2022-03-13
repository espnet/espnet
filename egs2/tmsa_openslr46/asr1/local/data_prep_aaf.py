import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-src', type=str)
parser.add_argument('--data-dest', type=str)
parser.add_argument('--data-kaldi', type=str)

args = parser.parse_args()

DATASET_DIR = args.data_src
DATA_DIR = args.data_kaldi
DEST_DIR = args.data_dest
# prepare the training data

TRAIN_SPEECH_DIR = os.path.join(DATASET_DIR, "speech/train")
TRAIN_SPEECH_CA16_DIR = os.path.join(TRAIN_SPEECH_DIR, "ca16")
TRAIN_SPEECH_YAOUNDE_DIR = os.path.join(TRAIN_SPEECH_DIR, "yaounde")

for speaker in os.listdir(TRAIN_SPEECH_CA16_DIR):
    for utterance in os.listdir(os.path.join(TRAIN_SPEECH_CA16_DIR, speaker)):
        src = os.path.join(os.path.join(TRAIN_SPEECH_CA16_DIR, speaker, utterance))
        dest = os.path.join(DEST_DIR, 'train', utterance)
        shutil.copy(src, dest)


utterance_types = ["read", "answers"]
for utt_type in utterance_types:
    for speaker in os.listdir(os.path.join(TRAIN_SPEECH_YAOUNDE_DIR, utt_type)):
        for utterance in os.listdir(os.path.join(TRAIN_SPEECH_YAOUNDE_DIR, utt_type, speaker)):
            src = os.path.join(TRAIN_SPEECH_YAOUNDE_DIR, utt_type, speaker, utterance)
            dest = os.path.join(DEST_DIR, 'train', utt_type + '_' + utterance)
            shutil.copy(src, dest)

# write the transcripts for training data

TRAIN_TRANSCRIPT_DIR = os.path.join(DATASET_DIR, "transcripts/train")

with open(os.path.join(DATA_DIR, 'train', 'text'), 'w') as f_text:
    with open(os.path.join(DATA_DIR, "train", "wav.scp"), 'w') as f_wav:
        with open(os.path.join(DATA_DIR, "train", "utt2spk"), 'w') as f_utt2spk:
            with open(os.path.join(DATA_DIR, "train", "spk2utt"), 'w') as f_spk2utt:
                with open(os.path.join(TRAIN_TRANSCRIPT_DIR, 'ca16_conv', 'transcripts.txt')) as f_trans:
                    for line in f_trans:
                        utt_id, text = line.split(' ', 1)
                        utt_id = utt_id[0: -4]
                        f_text.write(utt_id + " " + text)

                        wav_path = os.path.join('egs2/tmsa_openslr46/asr1', DEST_DIR, 'train', utt_id)
                        f_wav.write(utt_id + " " + wav_path + "\n")

                        f_utt2spk.write(utt_id+ " "+ utt_id + "\n")
                        f_spk2utt.write(utt_id + " " + utt_id + "\n")

                with open(os.path.join(TRAIN_TRANSCRIPT_DIR, 'ca16_read', 'conditioned.txt')) as f_trans:
                    for line in f_trans:
                        utt_id, text = line.split(' ', 1)
                        f_text.write(utt_id + " " + text)

                        wav_path = os.path.join('egs2/tmsa_openslr46/asr1', DEST_DIR, 'train', utt_id)
                        f_wav.write(utt_id + " " + wav_path + "\n")

                        f_utt2spk.write(utt_id + " " + utt_id + "\n")
                        f_spk2utt.write(utt_id + " " + utt_id + "\n")

                with open(os.path.join(TRAIN_TRANSCRIPT_DIR, 'yaounde', 'fn_text.txt')) as f_trans:
                    for line in f_trans:
                        file, text = line.split(' ', 1)
                        file = file.split('/')
                        if 'read' in file:
                            utt_id = 'read' + '_' + file[-1][0: -4]
                        else:
                            utt_id = 'answers' + '_' + file[-1][0: -4]

                        f_text.write(utt_id + " " + text)

                        wav_path = os.path.join('egs2/tmsa_openslr46/asr1', DEST_DIR, 'train', utt_id)
                        f_wav.write(utt_id + " " + wav_path + "\n")

                        f_utt2spk.write(utt_id + " " + utt_id + "\n")
                        f_spk2utt.write(utt_id + " " + utt_id + "\n")

# prepare the test data

TEST_SPEECH_DIR = os.path.join(DATASET_DIR, "speech/test")

for speaker in os.listdir(os.path.join(TEST_SPEECH_DIR, 'ca16')):
    for utterance in os.listdir(os.path.join(TEST_SPEECH_DIR, 'ca16', speaker)):
        src = os.path.join(TEST_SPEECH_DIR, 'ca16', speaker, utterance)
        dest = os.path.join(DEST_DIR, 'test', utterance)
        shutil.copy(src, dest)

# write the transcripts for test data

TEST_TRANSCRIPT_PATH = os.path.join(DATASET_DIR, "transcripts", "test", "ca16", "prompts.txt")

with open(os.path.join(DATA_DIR, 'test', 'text'), 'w') as f_text:
    with open(os.path.join(DATA_DIR, "test", "wav.scp"), 'w') as f_wav:
        with open(os.path.join(DATA_DIR, "test", "utt2spk"), 'w') as f_utt2spk:
            with open(os.path.join(DATA_DIR, "test", "spk2utt"), 'w') as f_spk2utt:
                with open(TEST_TRANSCRIPT_PATH, 'r') as f_trans:
                    for line in f_trans:
                        utt_id, text = line.split(' ', 1)
                        f_text.write(line)

                        wav_path = os.path.join('egs2/tmsa_openslr46/asr1', DEST_DIR, 'test', utt_id)
                        f_wav.write(utt_id + " " +  wav_path + "\n")

                        f_utt2spk.write(utt_id + " " + utt_id + "\n")
                        f_spk2utt.write(utt_id + " " + utt_id + "\n")

# prepare the validation data

VALID_SPEECH_DIR = os.path.join(DATASET_DIR, "speech/devtest")

for speaker in os.listdir(os.path.join(VALID_SPEECH_DIR, 'ca16')):
    for utterance in os.listdir(os.path.join(VALID_SPEECH_DIR, 'ca16', speaker)):
        src = os.path.join(VALID_SPEECH_DIR, 'ca16', speaker, utterance)
        dest = os.path.join(DEST_DIR, 'valid', utterance)
        shutil.copy(src, dest)


# write the transcripts for validation data

VALID_TRANSCRIPT_PATH = os.path.join(DATASET_DIR, "transcripts", "devtest", "ca16_read", "conditioned.txt")

with open(os.path.join(DATA_DIR, 'valid', 'text'), 'w') as f_text:
    with open(os.path.join(DATA_DIR, "valid", "wav.scp"), 'w') as f_wav:
        with open(os.path.join(DATA_DIR, "valid", "utt2spk"), 'w') as f_utt2spk:
            with open(os.path.join(DATA_DIR, "valid", "spk2utt"), 'w') as f_spk2utt:
                with open(VALID_TRANSCRIPT_PATH, 'r') as f_trans:
                    for line in f_trans:
                        utt_id, text = line.split(' ', 1)
                        f_text.write(line)

                        wav_path = os.path.join('egs2/tmsa_openslr46/asr1', DEST_DIR, 'test', utt_id)
                        f_wav.write(utt_id + " " + wav_path + "\n")

                        f_utt2spk.write(utt_id + " " + utt_id + "\n")
                        f_spk2utt.write(utt_id + " " + utt_id + "\n")

