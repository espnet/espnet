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
    
    def __len__(self):
        return len(self.dialogues)
    
    def __contains__(self, item):
        return item in self.dialogues

    def add_dialogue(self, name: str, dialogue: Dialogue):
        assert name not in self.dialogues
        self.dialogues[name] = dialogue.to_list()
    
    def convert_spoken_format(self, name, segments):
        if not name.endswith("_spoken"):
            return segments, []
        
        retval = list()
        uttid_content_pairs = list()

        # (1) If there is a system instruction, add it
        if segments[0][0] == "<system_prompt>":
            retval.append(segments.pop(0))
        
        # (2) Add an assistant speaker prompt
        retval.append(("<system_prompt>", "spk", f"{name}_spk"))

        # (3) loop over segment
        for idx, (role, modality, content) in enumerate(segments):
            role_str = role.lstrip("<").rstrip(">")
            utt_id = f"{name}_{idx}_{role_str}"

            if role == "<assistant_output>":
                retval.append((role, modality, content))
                retval.append((role, "codec_ssl", utt_id))
            elif role == "<user_input>":
                retval.append((role, "codec_ssl", utt_id))
                retval.append((role, modality, content))
            else:
                raise ValueError(f"unrecognized role: {role}")
            
            uttid_content_pairs.append((utt_id, content))
        
        return retval, uttid_content_pairs

    def dump_dataset(
        self, 
        output_dir, 
        pack_size: int = 20000, 
        rank: int = 1,
        convert_spoken_format: bool = False,
    ):
        output_dir = Path(output_dir)

        (output_dir / 'data').mkdir(parents=True, exist_ok=True)
        index_file = str(output_dir / 'data' / f'dialogue.{rank}')
        index_writer = open(index_file, 'w')

        example_ids = list(self.dialogues.keys())
        pack_idx = 0

        if convert_spoken_format:
            dialogues, uttid_content_pairs = dict(), list()
            for name, segments in self.dialogues.items():
                segments, uttid_content_pair = self.convert_spoken_format(
                    name, segments
                )
                dialogues[name] = segments
                uttid_content_pairs.extend(uttid_content_pair)
        else:
            dialogues, uttid_content_pairs = self.dialogues, []

        while pack_idx * pack_size < len(example_ids):
            start = pack_idx * pack_size
            end = min((pack_idx + 1) * pack_size, len(example_ids))
            pack = {key: dialogues[key] for key in example_ids[start: end]}

            pack_file = str(output_dir / 'data' / f'dialogue.{rank}.{pack_idx + 1}.json')
            
            for key in pack:
                index_writer.write(f"{key} {pack_file}\n")
             
            pack_writer = open(pack_file, 'wb')
            pack_writer.write(
                json.dumps(pack, indent=4, ensure_ascii=False, sort_keys=False).encode(
                    "utf_8"
                )
            )

            pack_idx += 1
        
        # (3) uttid - content pairs for further TTS simulation
        if convert_spoken_format > 0:
            simulation_writer = open(output_dir / 'data' / f'simu_text.{rank}', 'w')
            for uttid, content in uttid_content_pairs:
                simulation_writer.write(f"{uttid} {content}\n")
            
        # (3) dump ESPnet-SpeechLM style data.json
        # writer = open(output_dir / 'data.json', 'wb')
        # data_json = {
        #     "task": task,
        #     "vocabularies": [],
        #     "data_files": [f"{index_file},dialogue,dialogue_json"],
        #     "examples": list(self.dialogues.keys()),
        #     "num_examples": len(list(self.dialogues.keys())),
        # }
        # writer.write(
        #     json.dumps(data_json, indent=4, ensure_ascii=False, sort_keys=False).encode(
        #         "utf_8"
        #     )
        # )