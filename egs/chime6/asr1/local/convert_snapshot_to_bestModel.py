#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch


def main():
    model = torch.load(args.snapshot, map_location=torch.device("cpu"))["model"]
    torch.save(model, args.out)


def get_parser():
    parser = argparse.ArgumentParser(description='average models from snapshot')
    parser.add_argument("--snapshot", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    return parser
    

if __name__ == '__main__':
    args = get_parser().parse_args()
    main()

