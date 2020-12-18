#!/usr/bin/env python
import argparse
from distutils.version import LooseVersion
import warnings


def check(pytorch_version: str, cuda_version: str):
    # NOTE(kamo):  Supported cuda version is defined
    # as existing prebuilt binaries in
    # https://anaconda.org/pytorch/pytorch/files
    # You probably could perform pytorch with the cuda-version
    # if you built pytorch at local.
    maybe_supported = []
    # 1.7.0 or 1.7.1
    if LooseVersion("1.8") > LooseVersion(pytorch_version) >= LooseVersion("1.7"):
        supported = ["11.0", "10.2", "10.1", "9.2"]
    # 1.6.0
    elif LooseVersion(pytorch_version) >= LooseVersion("1.6"):
        supported = ["10.2", "10.1", "9.2"]
        # FIXME(kamo): 10.0 is not existing, but it seems to work in my environment
        maybe_supported = ["10.0"]
    # 1.5.0 or 1.5.1
    elif LooseVersion(pytorch_version) >= LooseVersion("1.5"):
        supported = ["10.2", "10.1", "9.2"]
        # FIXME(kamo): 10.0 is not existing, but it seems to work in my environment
        maybe_supported = ["10.0"]
    # 1.4.0
    elif LooseVersion(pytorch_version) >= LooseVersion("1.4"):
        supported = ["10.1", "10.0", "9.2"]
    # 1.3.0 or 1.3.1
    elif LooseVersion(pytorch_version) >= LooseVersion("1.3"):
        supported = ["10.1", "10.0", "9.2"]
    # 1.2.0
    elif LooseVersion(pytorch_version) >= LooseVersion("1.2"):
        supported = ["10.0", "9.2"]
    # 1.1.0
    elif LooseVersion(pytorch_version) >= LooseVersion("1.1"):
        supported = ["10.0", "9.0"]
    # 1.0.1
    elif LooseVersion(pytorch_version) >= LooseVersion("1.0.1"):
        supported = ["10.0", "9.0", "8.0"]
    # 1.0.0
    elif LooseVersion(pytorch_version) >= LooseVersion("1.0.0"):
        supported = ["10.0", "9.0", "8.0"]
    else:
        raise NotImplementedError(f"pytorch={pytorch_version}")

    for v in supported + maybe_supported:
        if cuda_version == v:
            print(v)
            if v in maybe_supported:
                warnings.warn(
                    f"pytorch={pytorch_version} with cuda={cuda_version} might not work."
                )
            break
    else:
        raise RuntimeError(
            f"Not compatible: pytorch={pytorch_version}, cuda={cuda_version}: "
            f"Supported cuda versions: {supported + maybe_supported}"
        )


def get_parser():
    parser = argparse.ArgumentParser(description="Check pytorch-cuda compatibility")
    parser.add_argument("pytorch_version")
    parser.add_argument("cuda_version")
    return parser


def main():
    parser = get_parser()
    check(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
