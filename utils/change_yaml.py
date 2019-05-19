import argparse
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--inyaml', type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('-o', '--outyaml', type=argparse.FileType('w'),
                        default=sys.stdout)
    parser.add_argument('-a', '--arg', action='append', default=[],
                        help="e.g -a a.b.c=4 -> {'a': {'b': {'c': 4}}}")
    args = parser.parse_args()

    indict = yaml.load(args.inyaml, Loader=yaml.Loader)
    if indict is None:
        indict = {}
    for arg in args.arg:
        if '=' in arg:
            key, value = arg.split('=')
            if value.strip() == '':
                pass
            else:
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
                    # If "=" is not included, it'll be deleted.
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
    yaml.dump(indict, args.outyaml, Dumper=yaml.Dumper)


if __name__ == '__main__':
    main()
