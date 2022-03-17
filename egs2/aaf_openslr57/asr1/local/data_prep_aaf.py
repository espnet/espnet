import os
import shutil
import argparse
import string

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

utterances = set()
with open(os.path.join(DATA_DIR, "train", "utt2spk"), 'w') as f_utt2spk:
    for speaker in os.listdir(TRAIN_SPEECH_CA16_DIR):
        for utterance in os.listdir(os.path.join(TRAIN_SPEECH_CA16_DIR, speaker)):
            src = os.path.join(os.path.join(TRAIN_SPEECH_CA16_DIR, speaker, utterance))
            dest = os.path.join(DEST_DIR, 'train', utterance)
            shutil.copy(src, dest)
            utt_id = utterance[0: -4]
            if utt_id in utterances:
                print(utt_id)
            utterances.add(utt_id)
            f_utt2spk.write(speaker + '_'+ utt_id + " " + speaker+"\n")

# write the transcripts for training data
TRAIN_TRANSCRIPT_DIR = os.path.join(DATASET_DIR, "transcripts/train")

with open(os.path.join(DATA_DIR, 'train', 'text'), 'w') as f_text:
    with open(os.path.join(DATA_DIR, "train", "wav.scp"), 'w') as f_wav:
        with open(os.path.join(TRAIN_TRANSCRIPT_DIR, 'ca16_conv', 'transcripts.txt'), 'r') as f_trans:
            for line in f_trans:
                utt_id, text = line.split(' ', 1)
                text = "".join([i.lower() for i in text if i not in string.punctuation])

                utt_id = utt_id[0: -4]
                speaker = utt_id.split('_')[2]
                if utt_id in utterances:
                    f_text.write(speaker + '_' + utt_id + " " + text)

                    wav_path = os.path.join(DEST_DIR, 'train', utt_id + '.wav')
                    f_wav.write(speaker + '_' + utt_id + " " + wav_path + "\n")

        with open(os.path.join(TRAIN_TRANSCRIPT_DIR, 'ca16_read', 'conditioned.txt')) as f_trans:
            for line in f_trans:
                utt_id, text = line.split(' ', 1)
                text = "".join([i.lower() for i in text if i not in string.punctuation])

                speaker = utt_id.split('_')[2]
                if utt_id in utterances:
                    f_text.write(speaker + '_' + utt_id + " " + text)

                    wav_path = os.path.join(DEST_DIR, 'train', utt_id + '.wav')
                    f_wav.write(speaker + '_' + utt_id + " " + wav_path + "\n")

# prepare the test data

TEST_SPEECH_DIR = os.path.join(DATASET_DIR, "speech/test")
utterances = set()


with open(os.path.join(DATA_DIR, "test", "utt2spk"), 'w') as f_utt2spk:
    for speaker in os.listdir(os.path.join(TEST_SPEECH_DIR, 'ca16')):
        for utterance in os.listdir(os.path.join(TEST_SPEECH_DIR, 'ca16', speaker)):
                src = os.path.join(TEST_SPEECH_DIR, 'ca16', speaker, utterance)
                dest = os.path.join(DEST_DIR, 'test', utterance)
                shutil.copy(src, dest)
                utt_id = utterance[0: -4]
                if utt_id in utterances:
                    print(utt_id)
                utterances.add(utt_id)
                f_utt2spk.write(speaker + '_'+ utt_id + " " + speaker+"\n")

# write the transcripts for test data

TEST_TRANSCRIPT_PATH = os.path.join(DATASET_DIR, "transcripts", "test", "ca16", "prompts.txt")

with open(os.path.join(DATA_DIR, 'test', 'text'), 'w') as f_text:
    with open(os.path.join(DATA_DIR, "test", "wav.scp"), 'w') as f_wav:
        with open(TEST_TRANSCRIPT_PATH, 'r') as f_trans:
            for line in f_trans:
                utt_id, text = line.split(' ', 1)
                text = "".join([i.lower() for i in text if i not in string.punctuation])
                speaker = '_'.join(utt_id.split('_')[:3])
                if utt_id in utterances:
                    f_text.write(speaker + '_' + utt_id + ' ' + text)

                    wav_path = os.path.join(DEST_DIR, 'test', utt_id + '.wav')
                    f_wav.write(speaker + '_' + utt_id + " " + wav_path + "\n")

# prepare the validation data

VALID_SPEECH_DIR = os.path.join(DATASET_DIR, "speech/devtest")
utterances = set()

with open(os.path.join(DATA_DIR, "valid", "utt2spk"), 'w') as f_utt2spk:
    for speaker in os.listdir(os.path.join(VALID_SPEECH_DIR, 'ca16')):
        for utterance in os.listdir(os.path.join(VALID_SPEECH_DIR, 'ca16', speaker)):
            src = os.path.join(VALID_SPEECH_DIR, 'ca16', speaker, utterance)
            dest = os.path.join(DEST_DIR, 'valid', utterance)
            shutil.copy(src, dest)
            utt_id = utterance[0: -4]
            if utt_id in utterances:
                print(utt_id)
            utterances.add(utt_id)
            f_utt2spk.write(speaker + '_'+ utt_id + " " + speaker+"\n")

# write the transcripts for validation data

VALID_TRANSCRIPT_PATH = os.path.join(DATASET_DIR, "transcripts", "devtest", "ca16_read", "conditioned.txt")

with open(os.path.join(DATA_DIR, 'valid', 'text'), 'w') as f_text:
    with open(os.path.join(DATA_DIR, "valid", "wav.scp"), 'w') as f_wav:
        with open(VALID_TRANSCRIPT_PATH, 'r') as f_trans:
            for line in f_trans:
                utt_id, text = line.split(' ', 1)
                text = "".join([i.lower() for i in text if i not in string.punctuation])
                speaker = utt_id.split('_')[2]
                if utt_id in utterances:
                    f_text.write(speaker + '_' + line)

                    wav_path = os.path.join(DEST_DIR, 'valid', utt_id + '.wav')
                    f_wav.write(speaker + '_' + utt_id + " " + wav_path + "\n")
