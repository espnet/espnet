#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging

from pathlib import Path
from datasets import load_dataset

from espnet2.utils.types import str2bool
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

logging.basicConfig(
    level="INFO",
    format=f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

def smoltalk_fn(
    sample,
    name: str = None,
):
    dialogue = Dialogue()
    # print('original sample: ', sample)
    for segment in sample['messages']:
        dialogue.add_segment(
            role=segment['role'],
            modality='text_bpe',
            content=segment['content']
        )

    return name, dialogue


HF_SFT_DATA = {
    "HuggingFaceTB/smoltalk": {
        # NOTE(Jinchuan): subsets and their processing methods.
        #  These methods are applied only when args.spoken_format is True.
        #  (1) original: keep original text format
        #  (2) skip: skip that subset
        #  (3) s2s: convert to speech-to-speech dialogue
        #  (4) s2t: convert to speech-to-text dialogue

        # NOTE(Jinchuan):
        # (1) Skip apigen-80k to avoid tool-call
        # (2) Skip longalign as it's too long
        # (3) Skip numina-cot-100k as it's too long and complicated
        # (4) Skip self-oss-instruct as it contains codes
        # (5) Skip smol-constraints as it requires text formatting
        # (6) Skip smol-rewrite as we don't need rewrite
        # (7) Skip smol-summarize as we don't need summarize
        "subsets_and_methods": {
            "apigen-80k": "skip",
            "everyday-conversations": "original",
            "explore-instruct-rewriting": "s2s",
            "longalign": "skip",
            "metamathqa-50k": "s2s",
            "numina-cot-100k": "skip",
            "openhermes-100k": "s2s",
            "self-oss-instruct": "skip",
            "smol-constraints": "skip",
            "smol-magpie-ultra": "s2s",
            "smol-rewrite": "skip",
            "smol-summarize": "skip",
            "systemchats-30k": "s2s",
        },
        'splits': ["test"],
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
        "--spoken_format",
        type=str2bool,
        default=False,
        help="If true, data is formulated in spoken format",
    )

    # Setups only apply for Gemini / LLM processing
    parser.add_argument(
        "--model_id",
        type=str,
        default="gemini-1.5-pro-002",
        # default="gemini-1.5-flash-002",
        help="model_id of Gemini series",
    )
    parser.add_argument(
        "--max_len_words",
        type=int,
        default=1000,
        help="maximum number of words in each example",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # (1) prepare dataset
    dataset = HF_SFT_DATA[args.input_hf_tag]
    subsets_and_methods = dataset['subsets_and_methods']
    splits = dataset['splits']
    parse_fn = dataset['parse_fn']

    subsets_methods_splits = [
        (subset, method, split)
        for (subset, method) in subsets_and_methods.items()
        for split in splits
    ]

    # (2) prepare LLM
    if args.spoken_format:
        from espnet2.speechlm.dialogue.gemini_utils import GeminiAPIInterface
        llm_model = GeminiAPIInterface(args.model_id, None)
    else:
        llm_model = None

    for subset, method, split in subsets_methods_splits:
        if subset not in ["systemchats-30k"]:
            continue

        # (1) load HF dataset and initialize ESPnet dialogue dataset
        ds = load_dataset(args.input_hf_tag, subset)[split]
        task = "audio_dialogue" if args.spoken_format else "text_dialogue"
        dialogue_dataset = DialogueDataset(task=task)

        # (2) loop to processing
        for idx, example in enumerate(ds):
            name = f"{subset}_{split}_{idx:05d}"
            name, dialogue = parse_fn(
                sample=example,
                name=name,
            )

            # (2.1) text dialogue processing
            dialogue_dataset.add_dialogue(name, dialogue)

            # (2.2) audio dialogue processing
            if args.spoken_format:
                if isinstance(method, tuple):
                    method, model_id = method
                else:
                    model_id = args.model_id

                if method in ["s2t", "s2s"]:
                    spoken_dialogue = llm_model.generate_spoken_dialogue(
                        name,
                        dialogue, 
                        prompt_method=method,
                        model_id=model_id,
                        max_len_words=args.max_len_words,
                    )
                elif method in ["original"]:
                    spoken_dialogue = dialogue
                elif method in ["skip"]:
                    continue
                else:
                    raise NotImplementedError(f"unrecognized processing method: {method}")
                
                if spoken_dialogue is not None:
                    logging.info(f"save spoken dialogue transcript: {name}_spoken")
                    dialogue_dataset.add_dialogue(name + "_spoken", spoken_dialogue)
            
            if idx > 20:
                break

        dialogue_dataset.dump_dataset(
            args.output_dir / f"{subset}_{split}",
            task=task,
        )

        logging.info(f'Done preparing {args.input_hf_tag}-{subset}-{split}')
        

if __name__ == "__main__":
    main()