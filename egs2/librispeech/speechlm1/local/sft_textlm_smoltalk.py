#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

from pathlib import Path
from datasets import load_dataset

from espnet2.utils.types import str2bool
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

def smoltalk_fn(
    sample,
    name: str = None,
):
    dialogue = Dialogue()
    for segment in sample['messages']:
        if segment['role'] == 'system':
            role = "<system_prompt>"
        elif segment['role'] == 'user':
            role = "<user_input>"
        elif segment['role'] == 'assistant':
            role = "<assistant_output>"

        dialogue.add_segment(
            role=role,
            modality='text_bpe',
            content=segment['content']
        )

    return name, dialogue

HF_SFT_DATA = {
    "HuggingFaceTB/smoltalk": {
        "subsets": [
            "apigen-80k",
            "everyday-conversations",
            "explore-instruct-rewriting",
            "longalign",
            "metamathqa-50k",
            "numina-cot-100k",
            "openhermes-100k",
            "self-oss-instruct",
            "smol-constraints",
            "smol-magpie-ultra",
            "smol-rewrite",
            "smol-summarize",
            "systemchats-30k",
        ],
        'splits': [
            "train", 
            "test",
        ],
        "parse_fn": smoltalk_fn
    }
}

def get_parser():
    parser = argparse.ArgumentParser(
        description="Build dialogue dataset from smoltalk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_hf_tag",
        type=str,
        default="HuggingFaceTB/smoltalk",
        help="input HuggingFace dataset tag",
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        required=True, 
        help="output directory of smoltalk dialogue dataset"
    )
    parser.add_argument(
        "--use_audio",
        type=str2bool,
        default=False,
        help="If true, also include audio",
    )
    parser.add_argument(
        "--audio_list",
        type=str,
        default=None,
        help="The audio list in <utterance-id> <content> format",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    subsets = HF_SFT_DATA[args.input_hf_tag]['subsets']
    splits = HF_SFT_DATA[args.input_hf_tag]['splits']
    parse_fn = HF_SFT_DATA[args.input_hf_tag]['parse_fn']
    subsets_splits = [(subset, split) for subset in subsets for split in splits]

    for subset, split in subsets_splits:
        ds = load_dataset(args.input_hf_tag, subset)[split]
        task = "audio_dialogue" if args.use_audio else "text_dialogue"
        dialogue_dataset = DialogueDataset(task=task)

        for idx, example in enumerate(ds):
            name = f"{subset}_{split}_{idx:05d}"
            name, dialogue = parse_fn(
                sample=example,
                name=name,
            )
            dialogue_dataset.add_dialogue(name, dialogue)
        
        dialogue_dataset.dump_dataset(
            args.output_dir / f"{subset}_{split}",
            task=task,
        )

        print(f'Done preparing {args.input_hf_tag}-{subset}-{split}')
        

if __name__ == "__main__":
    main()