#!/usr/bin/env python3

"""Script to check whether the installation is done correctly."""

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import importlib
import logging
import sys
import traceback

from distutils.version import LooseVersion


# NOTE: add the libraries which are not included in setup.py
MANUALLY_INSTALLED_LIBRARIES = [
    ("espnet", None),
    ("kaldiio", None),
    ("matplotlib", None),
    ("chainer", ("6.0.0")),
    # ("chainer_ctc", None),
    # ("warprnnt_pytorch", ("0.1")),
]

# NOTE: list all torch versions which are compatible with espnet
COMPATIBLE_TORCH_VERSIONS = (
    "0.4.1",
    "1.0.0",
    "1.0.1",
    "1.0.1.post2",
    "1.1.0",
    "1.2.0",
    "1.3.0",
    "1.3.1",
    "1.4.0",
    "1.5.0",
    "1.5.1",
    "1.6.0",
    "1.7.0",
    "1.7.1",
)


def main(args):
    """Check the installation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disable cuda-related tests",
    )
    parser.add_argument(
        "--no-cupy",
        action="store_true",
        default=False,
        help="Disable cupy test",
    )
    args = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"python version = {sys.version}")

    library_list = []
    if args.no_cuda:
        args.no_cupy = True

    if not args.no_cupy:
        library_list.append(("cupy", ("6.0.0")))

    # check torch installation at first
    try:
        import torch

        logging.info(f"pytorch version = {torch.__version__}")
        if torch.__version__ not in COMPATIBLE_TORCH_VERSIONS:
            logging.warning(f"{torch.__version__} is not tested. please be careful.")
    except ImportError:
        logging.warning("torch is not installed.")
        logging.warning("please try to setup again and then re-run this script.")
        sys.exit(1)

    # warpctc can be installed only for pytorch < 1.7
    # if LooseVersion(torch.__version__) < LooseVersion("1.7.0"):
    #     library_list.append(("warpctc_pytorch", ("0.1.1", "0.1.2", "0.1.3", "0.2.1")))

    library_list.extend(MANUALLY_INSTALLED_LIBRARIES)

    # check library availableness
    logging.info("library availableness check start.")
    logging.info("# libraries to be checked = %d" % len(library_list))
    is_correct_installed_list = []
    for idx, (name, version) in enumerate(library_list):
        try:
            importlib.import_module(name)
            logging.info("--> %s is installed." % name)
            is_correct_installed_list.append(True)
        except ImportError:
            logging.warning("--> %s is not installed.\n###### Raw Error ######\n%s#######################" % (name, traceback.format_exc()))
            is_correct_installed_list.append(False)

    # warp-rnnt was only tested and successfull with CUDA_VERSION=10.0
    # however the library installation is optional ("warp-transducer" is used by default)
    try:
        importlib.import_module("warp_rnnt")
        is_correct_installed_list.append(True)
        library_list.append(("warp_rnnt", ("0.4.0")))
        logging.info("--> warp_rnnt is installed")
    except ImportError:
        logging.info("--> warp_rnnt is not installed (optional). Setup again with "
                     "CUDA_VERSION=10.0 if you want to use it.")

    logging.info("library availableness check done.")
    logging.info(
        "%d / %d libraries are correctly installed."
        % (sum(is_correct_installed_list), len(library_list))
    )

    if len(library_list) != sum(is_correct_installed_list):
        logging.warning("please try to setup again and then re-run this script.")
        sys.exit(1)

    # check library version
    num_version_specified = sum(
        [True if v is not None else False for n, v in library_list]
    )
    logging.info("library version check start.")
    logging.info("# libraries to be checked = %d" % num_version_specified)
    is_correct_version_list = []
    for idx, (name, version) in enumerate(library_list):
        if version is not None:
            # Note: temp. fix for warprnnt_pytorch
            # not found version with importlib
            if name == "warprnnt_pytorch" or name == "warp_rnnt":
                import pkg_resources

                vers = pkg_resources.get_distribution(name).version
            else:
                vers = importlib.import_module(name).__version__
            if vers is not None:
                is_correct = vers in version
                if is_correct:
                    logging.info("--> %s version is matched (%s)." % (name, vers))
                    is_correct_version_list.append(True)
                else:
                    logging.warning(
                        "--> %s version is incorrect (%s is not in %s)."
                        % (name, vers, str(version))
                    )
                    is_correct_version_list.append(False)
            else:
                logging.info(
                    "--> %s has no version info, but version is specified." % name
                )
                logging.info("--> maybe it is better to reinstall the latest version.")
                is_correct_version_list.append(False)
    logging.info("library version check done.")
    logging.info(
        "%d / %d libraries are correct version."
        % (sum(is_correct_version_list), num_version_specified)
    )

    if sum(is_correct_version_list) != num_version_specified:
        logging.info("please try to setup again and then re-run this script.")
        sys.exit(1)

    # check cuda availableness
    if args.no_cuda:
        logging.info("cuda availableness check skipped.")
    else:
        logging.info("cuda availableness check start.")
        import chainer
        import torch

        try:
            assert torch.cuda.is_available()
            logging.info("--> cuda is available in torch.")
        except AssertionError:
            logging.warning("--> it seems that cuda is not available in torch.")
        try:
            assert torch.backends.cudnn.is_available()
            logging.info("--> cudnn is available in torch.")
        except AssertionError:
            logging.warning("--> it seems that cudnn is not available in torch.")
        try:
            assert chainer.backends.cuda.available
            logging.info("--> cuda is available in chainer.")
        except AssertionError:
            logging.warning("--> it seems that cuda is not available in chainer.")
        try:
            assert chainer.backends.cuda.cudnn_enabled
            logging.info("--> cudnn is available in chainer.")
        except AssertionError:
            logging.warning("--> it seems that cudnn is not available in chainer.")
        try:
            from cupy.cuda import nccl  # NOQA

            logging.info("--> nccl is installed.")
        except ImportError:
            logging.warning(
                "--> it seems that nccl is not installed. multi-gpu is not enabled."
            )
            logging.warning(
                "--> if you want to use multi-gpu, please install it and then re-setup."
            )
        try:
            assert torch.cuda.device_count() > 1
            logging.info(
                f"--> multi-gpu is available (#gpus={torch.cuda.device_count()})."
            )
        except AssertionError:
            logging.warning("--> it seems that only single gpu is available.")
            logging.warning("--> maybe your machine has only one gpu.")
        logging.info("cuda availableness check done.")

    logging.info("installation check is done.")


if __name__ == "__main__":
    main(sys.argv[1:])
