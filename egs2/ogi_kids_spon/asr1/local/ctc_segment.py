#ctc segmentation adapted from: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

import pandas as pd
import os
import re
from glob import glob

import librosa
import torchaudio
import torch
import soundfile as sf

from dataclasses import dataclass



@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words



def create_buckets(segments, ratio):
    bucket_duration = 25.0  # Maximum bucket duration in seconds
    buckets = []
    current_bucket = []
    current_duration = 0.0
    last_end = None

    for segment in segments:
        # Calculate duration of the current segment in seconds
        segment_duration = (segment.end - segment.start) * ratio
        
        # Calculate the gap duration from the last segment, if any
        if last_end is not None:
            gap_duration = (segment.start - last_end) * ratio
        else:
            gap_duration = 0.0

        # Check if adding this segment (with gap) would exceed the bucket duration
        if current_duration + gap_duration + segment_duration <= bucket_duration:
            current_bucket.append(segment)
            current_duration += gap_duration + segment_duration
        else:
            # Finalize the current bucket and start a new one
            buckets.append(current_bucket)
            current_bucket = [segment]
            current_duration = segment_duration
        last_end = segment.end  # Update last_end to the end of the current segment

    # Add the last bucket if it has any segments
    if current_bucket:
        buckets.append(current_bucket)

    return_intervals = []

    # Process the buckets into the desired interval format

    if not buckets:
        return return_intervals
    for bucket in buckets:
        if not bucket:
            continue
        first_start = bucket[0].start * ratio
        last_end = bucket[-1].end * ratio
        concatenated_text = " ".join([segment.label for segment in bucket]).strip()
        if concatenated_text:
            return_intervals.append((first_start, last_end, concatenated_text))
    
    return return_intervals

def clean_transcript(sentence, dictionary):
    sentence = re.sub(r'<[^>]*>', '', sentence)

    # Remove all asterisks
    sentence = re.sub(r'\*', '', sentence)
    sentence = re.sub(r'\([^)]*\)', '', sentence)
    sentence = re.sub(r'\[|\]', '', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)

    text = sentence.upper()
    text = text.replace(' ', '|')
    for t in text:
        if t not in dictionary:
            text = text.replace(t, '')
    return text

def return_transcript(text):
    text = text.replace(' ','')
    text = text.replace('|', ' ')
    text = text.lower()

    text = re.sub(r'\s+', ' ', text).strip()

    return text

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input" , type=str, required=True, help='Path to directory with input wav.scp and text files')
    parser.add_argument("--output", type=str, required=True, help='Path to directory to save the segmented audio files')
    parser.add_argument("--lists_dir", type=str, required=True, help='Path to directory with train, dev and test speaker lists')

    args = parser.parse_args()

    with open(f'{args.input}/wav.scp', 'r') as f:
        lines = f.readlines()
    
    with open(f'{args.input}/text', 'r') as g:
        lines2 = g.readlines()
    
    data = []
    for l1, l2 in zip(lines, lines2):
        utt_id, path = l1.strip().split(maxsplit=1)
        _, text = l2.strip().split(maxsplit=1)

        data.append((utt_id, path, text))
    
    df = pd.DataFrame(data, columns=['utt_id', 'path', 'transcript'])

    print(len(df), flush=True)

    df['clean transcript'] = ''
    df['duration'] = 0.0

    device = torch.device('cuda')

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()


    dictionary = {c: i for i, c in enumerate(labels)}
    print(dictionary, flush=True)



    for i, r in df.iterrows():

        df.at[i, 'clean transcript'] = clean_transcript(r['transcript'], dictionary)

        # Load audio file and calculate duration
        y, sr = librosa.load(r['path'])
        df.at[i, 'duration'] = librosa.get_duration(y=y, sr=sr)
        


    file_dict = []

    for c,r in df.iterrows():

        test_file = r['path']

        with torch.inference_mode():
            waveform, sr = torchaudio.load(test_file)
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        transcript = r['clean transcript']

        tokens = [dictionary[c] for c in transcript]


        trellis = get_trellis(emission, tokens)

        path = backtrack(trellis, emission, tokens)


        segments = merge_repeats(path)

        word_segments = merge_words(segments)

        ratio = waveform.size(1) / sr / trellis.size(0)
        intervals = create_buckets(segments, ratio)

        if not intervals:
            print(f'No intervals found for {test_file}', flush=True)
            continue

        audio, sr = librosa.load(test_file, sr=None)
        audio_length = len(audio) / sr
        buffer = 0.5

        for i, interval in enumerate(intervals):
            start, end, text = interval

            

            uniq_utt = test_file.split('/')[-1].split('.')[0]

            out_path = args.output + f'/wav/{uniq_utt}_{i}.wav'
            out_text_path = out_path.replace('.wav', '.txt')

            

            # Calculate new start and end times with a 0.5-second buffer
            
            new_start = max(0, start - buffer)
            new_end = min(audio_length, end + buffer)

            # Load the audio segment with the new start and end times
            y, sr = librosa.load(test_file, sr=None, offset=new_start, duration=new_end - new_start)

            text = return_transcript(text)

            if text.strip()=='':
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Write the audio segment to the output path
            sf.write(out_path, y, sr)

            with open(out_text_path, 'w') as f:
                f.write(text)

            new_fid = out_path.split('/')[-1].split('.')[0]

            file_dict.append([new_fid, out_path, text])
        

        if c % 100 == 0:
            print(f'Processed {c} files', flush = True)
        


    all_segs_dir = args.output + '/all_segs'
    dev_dir = args.output + '/dev'
    train_dir = args.output + '/train'
    test_dir = args.output + '/test'

    os.makedirs(all_segs_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    with open(all_segs_dir+'/wav.scp', 'w') as f:
        with open(all_segs_dir+'/text', 'w') as g:
            for fid, path, text in file_dict:
                f.write(f"{fid} {path}\n")
                g.write(f"{fid} {text}\n")


    new_df = pd.DataFrame(file_dict, columns=['fid', 'path', 'text'])

    new_df.to_csv(all_segs_dir+'/metadata.csv', index=False)


    og_dev_speakers = set()

    with open(args.lists_dir+ '/dev.list', 'r') as f:
        lines = f.readlines()

        for line in lines:
            fid = line.strip()

            og_dev_speakers.add(fid[:5])


    og_train_speakers = set()

    with open(args.lists_dir+ '/train.list', 'r') as f:
        lines = f.readlines()

        for line in lines:
            fid = line.strip()

            og_train_speakers.add(fid[:5])


    og_test_speakers = set()

    with open(args.lists_dir+ '/test.list', 'r') as f:
        lines = f.readlines()

        for line in lines:
            fid = line.strip()

            og_test_speakers.add(fid[:5])


    with open(all_segs_dir+'/wav.scp', 'r') as f:
        lines1 = f.readlines()

    with open(all_segs_dir+'/text', 'r') as g:
        lines2 = g.readlines()


    for l1, l2 in zip(lines1,lines2):

        utt_id, path = l1.split()


        if utt_id[:5] in og_dev_speakers:
            with open(args.output+'/dev/wav.scp', 'a') as f:
                f.write(f"{utt_id} {path}\n")
            with open(args.output+'/dev/text', 'a') as g:
                g.write(l2.strip() + '\n')

            with open(args.output+'/dev/utt.list', 'a') as f:
                f.write(utt_id + '\n')
        
        elif utt_id[:5] in og_train_speakers:
            with open(args.output+'/train/wav.scp', 'a') as f:
                f.write(f"{utt_id} {path}\n")
            with open(args.output+'/train/text', 'a') as g:
                g.write(l2.strip() + '\n')
            
            with open(args.output+'/train/utt.list', 'a') as f:
                f.write(utt_id + '\n')
        else:
            with open(args.output+'/test/wav.scp', 'a') as f:
                f.write(f"{utt_id} {path}\n")
            with open(args.output+'/test/text', 'a') as g:
                g.write(l2.strip() + '\n')
            
            with open(args.output+'/test/utt.list', 'a') as f:
                f.write(utt_id + '\n')

