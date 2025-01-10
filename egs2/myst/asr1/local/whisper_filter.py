#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The filtering rule is based on paper https://arxiv.org/pdf/2309.07927.pdf
# 1. Filter out the utterances with WER < threshold (50%)
# 2. Filter out the utterances with less than n (3) words  
# 3. Remove utterances longer than 30 seconds in training and development
# 4. Normalize text {<no_signal> digits into to the same format}

#NOTE: This requires a few additional packages (apart from those installed from the transformers installation procedure from espnet, name evaluate and datasets)

import os
import sys
import argparse
import torch
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor


import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset, Audio, concatenate_datasets

class WhisperDataset():
    def __init__(self, data_paths, data_args=None):
        self.data_paths = data_paths
        
        datasets = []
        for split in data_paths.keys():

            datasets.append(self.create_dataset(data_paths[split]))
            
        self.data = concatenate_datasets(datasets)
    
    def create_dataset(self, data_path):

        audio_dict = self._load_audio_path(data_path['scp_path'])
        label_dict = self._load_label(data_path['text_label'])
        assert len(audio_dict) == len(label_dict), "label and sample size mismatch"

        paired_dict = {"utt_id": [], "audio": [], "sentence": []}

        # prepare dictionary for huggingface dataset
        for i in range(len(audio_dict)):
            utt, audio_path = audio_dict[i]
            label = label_dict[utt]
            paired_dict["utt_id"].append(utt)
            paired_dict["audio"].append(audio_path)
            paired_dict["sentence"].append(label)
        
        dataset = Dataset.from_dict(paired_dict).cast_column("audio", Audio())
        
        return dataset
    
    def _load_audio_path(self, wav_path):
        
        audio_dict = []

        with open(wav_path, 'r') as fin:
            line = fin.readline()
            while line:
                line = line.strip().split(' ')
                utt = line[0]
                i = 1
                
                try:
                    while not os.path.exists(line[i]):
                        i += 1
                except:
                    print("SCP FILE Not valid!")
                    break
                
                audio_dict.append((utt, line[i]))
                line = fin.readline()

        print("Reading {} lines from {}".format(len(audio_dict), wav_path))

        return audio_dict
    
    def _load_label(self, lab_path):
        label_dict = dict()
        with open(lab_path, 'r') as fin:
            line = fin.readline()
            
            while line:
                line= line.strip().split(' ')
                label_dict[line[0]] = ' '.join(line[1:])    
                line = fin.readline()
        
        print("Reading {} lines from {}".format(len(label_dict), lab_path))
        return label_dict


def main():
    parser = argparse.ArgumentParser(description="loading whisper model to filter the utterance with low quality transcription")
    parser.add_argument("--wav_scp", required=True, type=str)
    parser.add_argument("--trn_scp", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--wer_threshold", default=0.5, type=float)
    parser.add_argument("--remove_n_words", required=True, default=3, type=int)
    parser.add_argument("--remove_long_dur", required=True, default=30, type=int)
    parser.add_argument("--saved_utt_list", required=True, type=str)
        
    args = parser.parse_args()

    data_path = {"data": {"scp_path": args.wav_scp, "text_label": args.trn_scp}}
    dataset = WhisperDataset(data_path).data

    processor = WhisperProcessor.from_pretrained(args.model, cache_dir="cached_whisper_models/")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, cache_dir="cached_whisper_models/").to("cuda")
    
    print("Model Loaded, Start Decoding and do filtering...")
    metric_wer = evaluate.load("wer")
    uttlist_writer = open(args.saved_utt_list, "w")
    uttlist_writer.flush()

    num_utt = 0
    for testdata in dataset:
        num_utt += 1
        audio = testdata["audio"]

        utt_id = testdata["utt_id"]

        print(f"Transcribing utterance {utt_id}...")

        if args.remove_long_dur > 0:
            audio_duration = len(audio["array"]) / audio["sampling_rate"]
            if audio_duration > args.remove_long_dur:
                print(f"Duration error in utt_id: {utt_id} with duration: {audio_duration}")
                continue 
        
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        reference = processor.tokenizer._normalize(testdata['sentence'])
 
        if len(reference.split(" ")) < args.remove_n_words:
            print(f"Num word error in utt_id: {utt_id} with ref: {reference}")
            continue

        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        wer = metric_wer.compute(references=[reference], predictions=[prediction])
        print(f"WER: {wer} for utt_id: {utt_id}")
        if wer > args.wer_threshold:
            print(f"WER error in utt_id: {utt_id}")
            continue

        if num_utt % 1000 == 0:
            print("Processed {} utterances out of {}".format(num_utt, len(dataset)), flush=True)

        uttlist_writer.write(testdata["utt_id"] + "\n")
    
    uttlist_writer.close()


if __name__ == "__main__":
    main()

