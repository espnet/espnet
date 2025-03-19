#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import random
from pathlib import Path
from typing import List

from espnet2.speechlm.definitions import MODALITIES
from espnet2.fileio.read_text import read_2columns_text

class Dialogue:
    def __init__(self, task):
        self.segments = list()
        self.task = task
    
    def add_segment(
        self,
        role: str,
        modality: str,
        target: bool,
        content: str,
    ):
        if role not in [None, "system", "user", "assistant"]:
            raise ValueError(
                f"Role can only be None, system, user or assistant: {role}"
            )
    
        if modality not in MODALITIES:
            raise ValueError(f"unrecognized modality: {modality}")

        self.segments.append([role, modality, target, content])
    
    def transform(self, task, name):
        # Same task, keep the same
        if self.task == task:
            return self.segments, []
        # add placeholder for speech
        elif self.task == "text_dialogue" and task == "audio_dialogue":
            return self.text_dialogue_to_audio_dialogue(name)
        else:
            raise NotImplementedError

    def text_dialogue_to_audio_dialogue(self, name):
        # NOTE(Jinchuan): this text-dialogue to audio-dialogue is for spoken dialogue
        # system SFT. Loss is applied to every segment except user input speech.
        # Since there is no speech in the text-dialogue, we temporarily put a 
        # placeholder to each speech for follow-up TTS simulation.

        # Each segment: (role, modality, is_target, content)
        segments = list()
        utt_text_pairs = list()
        for idx, (role, modality, content) in enumerate(self.segments):
            if role == "system":
                assert idx == 0, "System instruction should only in the begining"
                assert modality == "text_bpe"
                segments.append([role, modality, False, content])

                if task == "audio_dialogue":
                    utt_text_pairs.append((
                        f"{name}_speaker_prompt",
                        f"<placeholder for speaker prompt>",
                    ))
            
            elif role == "user":
                if task == "audio_dialogue" or task == "audio_text_dialogue":
                    segments.append([role, "codec_ssl", False, f"{name}_{role}_seg{idx}"])
                    utt_text_pairs.append((
                        f"{name}_{role}_seg{idx}",
                        content,
                    ))

                    segments.append([role, modality, True, content])
                else:
                    segments.append([role, modality, False, content])
            
            elif role == "assistant":
                segments.append([role, modality, True, content])

                if task == "audio_dialogue":
                    segments.append([role, "codec_ssl", True, f"{name}_{role}_seg{idx}"])
                    utt_text_pairs.append((
                        f"{name}_{role}_seg{idx}",
                        content,
                    ))
            
            else:
                raise ValueError(f"Unrecognized role {role}")
        
        return segments, utt_text_pairs


    
    def to_str(self):
        string = ""
        for segment in self.segments:
            role, _, content = segment
            string += f"**{role}**: {content}\n"

        return string.strip()

class DialogueDataset:
    def __init__(self, task):

        if task not in ["text_dialogue", "audio_dialogue", "vision_dialogue"]:
            raise ValueError("dialogue support: text, audio, vision")
        self.task = task

        self.dialogues = dict()
    
    def __len__(self):
        return len(self.dialogues)
    
    def __contains__(self, item):
        return item in self.dialogues

    def add_dialogue(self, name: str, dialogue: Dialogue):
        assert name not in self.dialogues
        self.dialogues[name] = dialogue

    def dump_dataset(
        self, 
        output_dir, 
        pack_size: int = 20000, 
        rank: int = 1,
        assistant_prompt_list: str = None,
        user_prompt_list: str = None,
    ):
        output_dir = Path(output_dir)

        if assistant_prompt_list is not None and user_prompt_list is not None:
            assistant_prompt_list = read_2columns_text(assistant_prompt_list)
            assistant_prompt_list = list(assistant_prompt_list.values())
            user_prompt_list = read_2columns_text(user_prompt_list)
            user_prompt_list = list(user_prompt_list.values())
        else:
            assistant_prompt_list, user_prompt_list = None, None

        (output_dir / 'data').mkdir(parents=True, exist_ok=True)
        index_file = str(output_dir / 'data' / f'dialogue.{rank}')
        index_writer = open(index_file, 'w')

        example_ids = list(self.dialogues.keys())
        pack_idx = 0
        all_utt_text_pairs = list()
        while pack_idx * pack_size < len(example_ids):
            start = pack_idx * pack_size
            end = min((pack_idx + 1) * pack_size, len(example_ids))
            pack_idx += 1

            # (1) dump original dialogue data
            # NOTE(Jinchuan): If, so far the dialogue are still with format text_dialogue
            # We convert it to audio_dialogue or audio_text_dialogue format when
            # self.task is that format. The utt_text_pairs are paired utterance-id and
            # text for TTS simulation.
            pack, all_utt_text_pairs = dict(), dict()
            for key in example_ids[start: end]:
                dialogue = self.dialogues[key]
                dialogue_segments, utt_text_pairs = dialogue.transform(self.task, key)
                if not check_valid_segments(dialogue_segments):
                    continue
                pack[key] = dialogue_segments
                all_utt_text_pairs.update({
                    x[0]: x[1] for x in utt_text_pairs
                })

            pack_file = str(output_dir / 'data' / f'dialogue_rank{rank}_pack{pack_idx}.json')

            for key in pack:
                index_writer.write(f"{key} {pack_file}\n")

            pack_writer = open(pack_file, 'wb')
            pack_writer.write(
                json.dumps(pack, indent=4, ensure_ascii=False, sort_keys=False).encode(
                    "utf_8"
                )
            )

            # (2) dump ESPnet-SLM style TTS dataset for TTS simulation
            if self.task == "audio_dialogue" and len(all_utt_text_pairs) > 0:
                assert assistant_prompt_list is not None
                assert user_prompt_list is not None

                tts_dir = output_dir / 'tts_simulation' / f"rank{rank}_pack{pack_idx}"
                tts_dir.mkdir(parents=True, exist_ok=True)

                wav_scp_writer = open(tts_dir / 'wav.scp', 'w')
                utt2spk_writer = open(tts_dir / 'utt2spk', 'w')
                text_writer = open(tts_dir / 'text', 'w')

                for key, segments in pack.items():
                    assistant_prompt = random.choice(assistant_prompt_list)
                    user_prompt = random.choice(user_prompt_list)

                    for role, modality, _, content in segments:
                        
                        if modality == "codec_ssl":
                            text = all_utt_text_pairs[content]
                            # target wave is just a placeholder, will not be in real use
                            wav_scp_writer.write(f"{content} {assistant_prompt}\n")
                            text_writer.write(f"{content} {text}\n")
                            if role == "user":
                                utt2spk_writer.write(f"{content} {user_prompt}\n")
                            elif role == "assistant":
                                utt2spk_writer.write(f"{content} {assistant_prompt}\n")
                            else:
                                raise ValueError(f"Unrecognized role: {role}")

def check_valid_segments(segments):
    for role, modality, target, content in segments:
        if content.strip() == "":
            return False
    
    return True
