#!/usr/bin/env python3

import argparse
import os

import yaml


def get_parser():
    parser = argparse.ArgumentParser(
        description="get a specified attribute from a YAML file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inyaml")
    return parser


def main():
    args = get_parser().parse_args()
    with open(args.inyaml, "r") as f:
        indict = yaml.load(f, Loader=yaml.Loader)

    if "token_list" not in indict:
        raise AttributeError("token_list is not found in config.")

    token_list = os.path.dirname(args.inyaml) + "/tokens.txt"
    with open(token_list, "w") as f:
        for token in indict["token_list"]:
            f.write(f"{token}\n")


if __name__ == "__main__":
    main()
