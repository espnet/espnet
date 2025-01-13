#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json

from pathlib import Path
from typing import List

from espnet2.speechlm.definitions import MODALITIES

class Dialogue:
    def __init__(self):
        self.segments = list()
    
    def add_segment(
        self,
        role: str,
        modality: str,
        content: str,
    ):
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Role can only be system, user or assistant: {role}")
    
        if modality not in MODALITIES:
            raise ValueError(f"unrecognized modality: {modality}")

        self.segments.append([role, modality, content])
    
    def to_list(self):
        retval = []

        for role, modality, content in self.segments:
            # Change role to a ESPnet-SLM special token
            if role == "system":
                role = "<system_prompt>"
            elif role == "user":
                role = "<user_input>"
            elif role == "assistant":
                role = "<assistant_output>"
            else:
                raise ValueError(f"Invalid role: {role}")
            
            retval.append([role, modality, content])

        return retval
    
    def to_str(self):
        string = ""
        for segment in self.segments:
            role, _, content = segment
            string += f"{role}: {content}\n"

        return string.strip()

class DialogueDataset:
    def __init__(self, task):

        if task not in ["text_dialogue", "audio_dialogue"]:
            raise ValueError("task can only be text_dialogue or audio_dialogue")
        self.task = task

        self.dialogues = dict()

    def add_dialogue(self, name: str, dialogue: Dialogue):
        assert name not in self.dialogues
        self.dialogues[name] = dialogue.to_list()
    
    def dump_dataset(self, output_dir, task, pack_size: int = 20000):
        output_dir = Path(output_dir)

        (output_dir / 'index_files').mkdir(parents=True, exist_ok=True)
        (output_dir / 'data').mkdir(parents=True, exist_ok=True)
        index_file = str(output_dir / 'index_files' / 'dialogue')
        index_writer = open(index_file, 'w')

        example_ids = list(self.dialogues.keys())
        pack_idx = 0

        while pack_idx * pack_size < len(example_ids):
            start = pack_idx * pack_size
            end = min((pack_idx + 1) * pack_size, len(example_ids))
            pack = {key: self.dialogues[key] for key in example_ids[start: end]}
            pack_file = str(output_dir / 'data' / f'dialogue.{pack_idx + 1}.json')
            
            for key in pack:
                index_writer.write(f"{key} {pack_file}\n")
             
            pack_writer = open(pack_file, 'wb')
            pack_writer.write(
                json.dumps(pack, indent=4, ensure_ascii=False, sort_keys=False).encode(
                    "utf_8"
                )
            )

            pack_idx += 1
        
        # (3) dump ESPnet-SpeechLM style data.json
        writer = open(output_dir / 'data.json', 'wb')
        data_json = {
            "task": task,
            "vocabularies": [],
            "data_files": [f"{index_file},dialogue,dialogue_json"],
            "examples": list(self.dialogues.keys()),
            "num_examples": len(list(self.dialogues.keys())),
        }
        writer.write(
            json.dumps(data_json, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )
        

        