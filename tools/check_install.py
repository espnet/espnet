#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import importlib
import logging
import sys


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cupy', action='store_true', default=False,
                        help='Disable CUPY tests')
    args = parser.parse_args(args)

    # you should add the libraries which are not included in setup.py
    MANUALLY_INSTALLED_LIBRARIES = [
        ('espnet', None),
        ('kaldiio', None),
        ('matplotlib', None),
        ('torch', ("0.4.1", "1.0.0", "1.0.1.post2")),
        ('chainer', ("6.0.0")),
        ('chainer_ctc', None),
        ('warpctc_pytorch', ("0.1.1")),
        ('warprnnt_pytorch', ("0.1.1"))
    ]

    if not args.no_cupy:
        MANUALLY_INSTALLED_LIBRARIES.append(('cupy', ("6.0.0")))

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s")

    logging.info("python version = " + sys.version)

    library_list = []
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
            logging.warning("--> %s is not installed." % name)
            is_correct_installed_list.append(False)
    logging.info("library availableness check done.")
    logging.info("%d / %d libraries are correctly installed." % (
        sum(is_correct_installed_list), len(library_list)))

    if len(library_list) != sum(is_correct_installed_list):
        logging.info("please try to setup again and then re-run this script.")
        sys.exit(1)

    # check library version
    num_version_specified = sum([True if v is not None else False for n, v in library_list])
    logging.info("library version check start.")
    logging.info("# libraries to be checked = %d" % num_version_specified)
    is_correct_version_list = []
    for idx, (name, version) in enumerate(library_list):
        if version is not None:
            # Note: temp. fix for warprnnt_pytorch
            # not found version with importlib
            if name == "warprnnt_pytorch":
                import pkg_resources
                vers = pkg_resources.get_distribution(name).version
            else:
                vers = importlib.import_module(name).__version__
            if vers != None:
                is_correct = vers in version
                if is_correct:
                    logging.info("--> %s version is matched." % name)
                    is_correct_version_list.append(True)
                else:
                    logging.warning("--> %s version is not matched (%s is not in %s)." % (
                        name, lib.__version__, str(version)))
                    is_correct_version_list.append(False)
            else:
                logging.info("--> %s has no version info, but version is specified." % name)
                logging.info("--> maybe it is better to reinstall the latest version.")
                is_correct_version_list.append(False)
    logging.info("library version check done.")
    logging.info("%d / %d libraries are correct version." % (
        sum(is_correct_version_list), num_version_specified))

    if sum(is_correct_version_list) != num_version_specified:
        logging.info("please try to setup again and then re-run this script.")
        sys.exit(1)

    # check cuda availableness
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
        logging.warning("--> it seems that nccl is not installed. multi-gpu is not enabled.")
        logging.warning("--> if you want to use multi-gpu, please install it and then re-setup.")
    try:
        assert torch.cuda.device_count() > 1
        logging.info("--> multi-gpu is available (#gpus = %d)." % torch.cuda.device_count())
    except AssertionError:
        logging.warning("--> it seems that only single gpu is available.")
        logging.warning('--> maybe your machine has only one gpu.')
    logging.info("cuda availableness check done.")


if __name__ == '__main__':
    main(sys.argv[1:])
