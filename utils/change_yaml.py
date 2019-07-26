#!/usr/bin/env python
import argparse
from pathlib import Path

import yaml


def get_parser():
    parser = argparse.ArgumentParser(
        description='change specified attributes of a YAML file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inyaml', nargs='?')
    parser.add_argument('-o', '--outyaml')
    parser.add_argument('-a', '--arg', action='append', default=[],
                        help="e.g -a a.b.c=4 -> {'a': {'b': {'c': 4}}}")
    parser.add_argument('-d', '--delete', action='append', default=[],
                        help='e.g -d a -> "a" is removed from the input yaml')
    return parser


def main():
    args = get_parser().parse_args()

    if args.inyaml is None:
        indict = {}
    else:
        with open(args.inyaml, 'r') as f:
            indict = yaml.load(f, Loader=yaml.Loader)
        if indict is None:
            indict = {}

    if args.outyaml is None:
        # Auto naming from arguments
        eles = []
        if args.inyaml is not None:
            p = Path(args.inyaml)
            eles.append(str(p.parent / p.stem))

        table = str.maketrans('{}[]()', '%%__--', ' |&;#*?~"\'\\')
        for arg in args.delete:
            value = arg.translate(table)
            eles.append('del-' + value)
        for arg in args.arg:
            if '=' not in arg:
                raise RuntimeError(f'"{arg}" does\'t include "="')
            key, value = arg.split('=')
            key = key.translate(table)
            value = value.translate(table)
            eles.append(key + value)

        outyaml = '_'.join(eles)
        if outyaml == '':
            outyaml = 'config'
        outyaml += '.yaml'
        if args.inyaml == outyaml:
            p = Path(args.outyaml)
            outyaml = p.parent / (p.stem + '.2' + p.suffix)
    else:
        outyaml = args.outyaml

    for arg in args.delete + args.arg:
        if '=' in arg:
            key, value = arg.split('=')
            if not value.strip() == '':
                value = yaml.load(value, Loader=yaml.Loader)
        else:
            key = arg
            value = None

        keys = key.split('.')
        d = indict
        for idx, k in enumerate(keys):
            if idx == len(keys) - 1:
                if isinstance(d, (tuple, list)):
                    k = int(k)
                    if k >= len(d):
                        d += type(d)(None for _ in range(k - len(d) + 1))
                if value is not None:
                    d[k] = value
                else:
                    del d[k]
            else:
                if isinstance(d, (tuple, list)):
                    k = int(k)
                    if k >= len(d):
                        d += type(d)(None for _ in range(k - len(d) + 1))
                elif isinstance(d, dict):
                    if k not in d:
                        d[k] = {}
                if not isinstance(d[k], (dict, tuple, list)):
                    d[k] = {}
                d = d[k]
    with open(outyaml, 'w') as f:
        yaml.dump(indict, f, Dumper=yaml.Dumper)
    print(outyaml)


if __name__ == '__main__':
    main()
