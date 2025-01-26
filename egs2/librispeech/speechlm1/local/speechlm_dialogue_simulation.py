#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
As of Jan 2025, create supervised fine-tuning (SFT) data for SpeechLM

*** Text-Only fine-tuning, use the original data of the following datasets.
    Note, 

SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
  * This is for instruction following / task completion
  * around 1M samples
SODA: https://huggingface.co/datasets/allenai/soda
  * Role play-like conversation, very short (250 tokens)
  * Have some emotional labels, better for future exploration
  * around 1.5M samples
ultrachat_200k: 
  * Daily conversation without role, kind of long (1200 tokens)
  * Have HFRL data, better for future exploration

*** Audio-In-Audio-Out data simulation: We rely on TTS to simulate dialogue dataset.
  * We discard the ultrachat dataset as it is too long
  * We use the full SODA dataset
  * We use some subsets of SmolTalk; others are not suitable
    for speech application.
    ---------------------------------------
    ** apigen-80k: skip, to avoid tool-call
    ** longalign: skip, too long
    ** numina-cot-100k: skip, too long, CoT
    ** self-oss-instruct: skip, contains codes
    ** smol-constraints: skip, contains text formatting
    ** smol-rewrite: skip, don't need rewrite
    ** smol-summarize: skip, don't need summarize
    ---------------------------------------
"""


import argparse
import logging

from pathlib import Path
from datasets import load_dataset

from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

logging.basicConfig(
    level="INFO",
    format=f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def smoltalk_fn(
    sample,
    name,
):
    dialogue = Dialogue("text_dialogue")
    for segment in sample["messages"]:
        dialogue.add_segment(
            role=segment["role"], modality="text_bpe", content=segment["content"]
        )

    return name, dialogue

def soda_fn(
    sample,
    name,
):
    """
    In SODA dataset, there is always a narattive to provide the content.
    We use that content as system instruction.
    The dialogue is between two speakers. We assume the first speaker
    is the user and the second speaker is the assistant. We add this
    reminder in the system instruction.
    """

    dialogue = Dialogue("text_dialogue")

    # Add the narrative description
    narrative = sample['narrative']
    user_name, assistant_name = sample["speakers"][:2]
    role_str = f"This is a conversation between {user_name} and {assistant_name}. As the assistant, your role is {assistant_name}."
    narrative = role_str + "\n" + narrative
    dialogue.add_segment(
        role="system", 
        modality="text_bpe", 
        content=narrative
    )

    # add the dialogue segments
    for idx, content in enumerate(sample["dialogue"]):
        if idx % 2 == 0:
            role = "user"
        else:
            role = "assistant"
        dialogue.add_segment(
            role=role,
            modality="text_bpe",
            content=content
        )

    return name, dialogue

def ultrachat_fn(
    sample,
    name,
):
    dialogue = Dialogue("text_dialogue")

    for segment in sample["messages"]:
        dialogue.add_segment(
            role=segment['role'],
            modality="text_bpe",
            content=segment['content']
        )

    return name, dialogue

# NOTE(Jinchuan): for each subset and corresponding task,
# we assign the processing method.
#     Original: Keep the original text
#     llm: use LLM (e.g., Gemini) to process
#     skip: skip processing this subset.
HF_SFT_DATA = {
    "HuggingFaceTB/smoltalk": {
        "subsets_and_methods": {
            "apigen-80k": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "everyday-conversations": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "explore-instruct-rewriting": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "longalign": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "metamathqa-50k": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "numina-cot-100k": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "openhermes-100k": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "self-oss-instruct": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "smol-constraints": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "smol-magpie-ultra": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "smol-rewrite": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "smol-summarize": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
            "systemchats-30k": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            },
        },
        "splits": ["train", "test"],
        "parse_fn": smoltalk_fn,
    },
    "allenai/soda":{
        "subsets_and_methods": {
            "default": {
                "text_dialogue": "original",
                "audio_dialogue": "original", 
                "audio_text_dialogue": "skip",
            }
        },
        "splits": ["train", "validation", "test"],
        "parse_fn": soda_fn,
    },
    "HuggingFaceH4/ultrachat_200k":{
        "subsets_and_methods": {
            "default": {
                "text_dialogue": "original",
                "audio_dialogue": "skip",
                "audio_text_dialogue": "skip",
            }
        },
        "splits": ["train_sft", "test_sft"],
        "parse_fn": ultrachat_fn,
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
        choices=[
            "HuggingFaceTB/smoltalk",
            "allenai/soda",
            "HuggingFaceH4/ultrachat_200k",
        ],
        help="input HuggingFace dataset tag",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="output directory of smoltalk dialogue dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text_dialogue",
        choices=[
            "text_dialogue",  # Text-In-Text-Out
            "audio_dialogue",  # Audio-In-Audio-Out
            "audio_text_dialogue",  # Audio-In-Text-Out
        ],
        help="Dialogue task type",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="rank of slurm multiprocessing",
    )

    # Setups only apply for Gemini / LLM processing
    parser.add_argument(
        "--model_id",
        type=str,
        # default="gemini-1.5-pro-002",
        default="gemini-1.5-flash-002",
        help="model_id of Gemini series",
    )
    parser.add_argument(
        "--max_len_words",
        type=int,
        default=1000,
        help="maximum number of words in each example",
    )

    # TTS related
    parser.add_argument(
        "--assistant_prompt_list",
        type=Path,
        default=None,
        help="Speaker prompt to generate assistant output speech",
    )
    parser.add_argument(
        "--user_prompt_list",
        type=Path,
        default=None,
        help="Speaker prompt to generate user input speech",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # (1) prepare dataset
    logging.info(f"processing {args.input_hf_tag} for {args.task}")
    dataset = HF_SFT_DATA[args.input_hf_tag]
    subsets_and_methods = dataset["subsets_and_methods"]
    splits = dataset["splits"]
    parse_fn = dataset["parse_fn"]

    subsets_methods_splits = [
        (subset, method[args.task], split)
        for (subset, method) in subsets_and_methods.items()
        for split in splits
    ]

    # (2) prepare LLM
    if args.task != "text_dialogue":
        from espnet2.speechlm.dialogue.gemini_utils import GeminiAPIInterface

        llm_model = GeminiAPIInterface(args.model_id, None)
    else:
        llm_model = None

    for subset, method, split in subsets_methods_splits:
        if method == "skip":
            logging.info(f"Skip processing {args.input_hf_tag}-{subset}-{split} for {args.task}")
            continue
        else:
            logging.info(f"processing {args.input_hf_tag}-{subset}-{split} with method {method}")

        # (1) load HF dataset and initialize ESPnet dialogue dataset
        ds = load_dataset(args.input_hf_tag, subset)[split]
        input_hf_tag = args.input_hf_tag.replace("/", "-")
        dialogue_dataset = DialogueDataset(task=args.task)

        # (2) loop to processing
        for idx, example in enumerate(ds):
            name = f"{input_hf_tag}_{subset}_{split}_{idx:08d}"
            name, dialogue = parse_fn(
                sample=example,
                name=name,
            )

            if method == "original":
                dialogue = dialogue
            elif method == "llm":
                dialogue = llm_model.generate_spoken_dialogue(
                    name,
                    dialogue,
                    task=args.task,
                    model_id=args.model_id,
                    max_len_words=args.max_len_words,
                )
            else:
                raise ValueError(f"Cannot recognize method {method}")

            # (2.1) text dialogue processing
            dialogue_dataset.add_dialogue(f"{name}_{args.task}", dialogue)

        dialogue_dataset.dump_dataset(
            args.output_dir / f"{input_hf_tag}_{subset}_{split}",
            rank=args.rank,
            assistant_prompt_list=args.assistant_prompt_list,
            user_prompt_list=args.user_prompt_list,
        )

        logging.info(f"Done preparing {args.input_hf_tag}-{subset}-{split}")


if __name__ == "__main__":
    main()
