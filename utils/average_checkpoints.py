#!/usr/bin/env python
import argparse
import os


def main():
    last = sorted(args.snapshots, key=os.path.getmtime)
    last = last[-args.num:]
    print("average over", last)
    avg = None

    if args.backend == 'pytorch':
        import torch
        # sum
        for path in last:
            states = torch.load(path, map_location=torch.device("cpu"))["model"]
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    avg[k] += states[k]

        # average
        for k in avg.keys():
            if avg[k] is not None:
                avg[k] /= args.num

        torch.save(avg, args.out)
    elif args.backend == 'chainer':
        import numpy as np
        # sum
        for path in last:
            states = np.load(path)
            if avg is None:
                keys = [x.split('main/')[1] for x in states if 'model' in x]
                avg = dict()
                for k in keys:
                    avg[k] = states['updater/model:main/{}'.format(k)]
            else:
                for k in keys:
                    avg[k] += states['updater/model:main/{}'.format(k)]
        # average
        for k in keys:
            if avg[k] is not None:
                avg[k] /= args.num
        np.savez_compressed(args.out, **avg)
        os.rename('{}.npz'.format(args.out), args.out)  # numpy save with .npz extension
    else:
        raise ValueError('Incorrect type of backend')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--backend", default='chainer', type=str)
    args = parser.parse_args()
    main()
