#!/usr/bin/env python3
import argparse

import yaml


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inyaml')
    parser.add_argument('attr', help='foo.bar will access yaml.load(inyaml)["foo"]["bar"]')
    args = parser.parse_args()
    with open(args.inyaml, 'r') as f:
        indict = yaml.load(f, Loader=yaml.Loader)

    try:
        for attr in args.attr.split('.'):
            if attr.isdigit():
                attr = int(attr)
            indict = indict[attr]
        print(indict)
    except KeyError:
        # print nothing
        # sys.exit(1)
        pass


if __name__ == '__main__':
    main()
