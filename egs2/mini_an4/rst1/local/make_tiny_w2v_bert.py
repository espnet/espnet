#!/usr/bin/env python3
"""Write a tiny, randomly initialised w2v-BERT 2.0 for the mini_an4 recipe.

facebook/w2v-bert-2.0 (580M parameters) is far too large for a CPU test and has
no official tiny counterpart, so this saves a model with the same architecture
and input (160-dim stacked fbank, as rst_model computes it) in Hugging Face
format. The debug configs load it with ``model_tag: data/tiny_w2v_bert``. Apart
from the sizes, the configuration follows facebook/w2v-bert-2.0.
"""

import argparse

import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(cmd=None):
    args = get_parser().parse_args(cmd)
    torch.manual_seed(args.seed)
    config = Wav2Vec2BertConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=2 * args.hidden_size,
        feature_projection_input_dim=160,
        conv_depthwise_kernel_size=3,
        apply_spec_augment=False,
    )
    Wav2Vec2BertModel(config).save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
