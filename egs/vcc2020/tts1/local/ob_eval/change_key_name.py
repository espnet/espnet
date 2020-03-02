#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np
import torch
import collections


def main():
    
    # load params
    states = torch.load(args.model, map_location=torch.device("cpu"))
    new_states = collections.OrderedDict((k.replace("input_layer", "embed") if "input_layer" in k else k, v) for k, v in states.items())
    new_states = collections.OrderedDict((k.replace("encoder.norm", "encoder.after_norm") if "encoder.norm" in k else k, v) for k, v in new_states.items())
    new_states = collections.OrderedDict((k.replace("decoder.output_norm", "decoder.after_norm") if "decoder.output_norm" in k else k, v) for k, v in new_states.items())

    torch.save(new_states, args.out)


def get_parser():
    parser = argparse.ArgumentParser(description='change names in a model file')
    parser.add_argument("model", type=str)
    parser.add_argument("out", type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main()
