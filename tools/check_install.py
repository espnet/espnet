#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import importlib
import logging
import sys

# you should add the libraries which are not included in setup.py
MANUALLY_INSTALLED_LIBRARIES = [
    ('espnet', None),
    ('kaldi_io_py', None),
    ('matplotlib', None),
    ('torch', "0.4.1"),
    ('chainer', "5.0.0"),
    ('cupy', "5.0.0"),
    ('chainer_ctc', None),
    ('warpctc_pytorch', "0.1.1")
]

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
        logging.warn("--> %s is not installed." % name)
        is_correct_installed_list.append(False)
logging.info("library availableness check done.")
logging.info("%d / %d libraries are correctly installed." % (
    sum(is_correct_installed_list), len(library_list)))

# check library version
if len(library_list) != sum(is_correct_installed_list):
    logging.info("please try to setup again and then re-run this script.")
    sys.exit(1)
else:
    num_version_specified = sum([True if v is not None else False for n, v in library_list])
    logging.info("library version check start.")
    logging.info("# libraries to be checked = %d" % num_version_specified)
    is_correct_version_list = []
    for idx, (name, version) in enumerate(library_list):
        if version is not None:
            try:
                lib = importlib.import_module(name)
                if hasattr(lib, "__version__"):
                    assert lib.__version__ == version
                    logging.info("--> %s version is matched." % name)
                    is_correct_version_list.append(True)
                else:
                    logging.info("--> %s has no version info, but version is specified." % name)
                    logging.info("--> maybe it is better to reinstall the latest version.")
                    is_correct_version_list.append(False)
            except AssertionError:
                logging.warn("--> %s version is not matched (%s==%s)." % (name, lib.__version__, version))
                is_correct_version_list.append(False)
    logging.info("library version check done.")
    logging.info("%d / %d libraries are correct version." % (
        sum(is_correct_version_list), num_version_specified))

# check cuda availableness
if sum(is_correct_version_list) != num_version_specified:
    logging.info("please try to setup again and then re-run this script.")
    sys.exit(1)
else:
    logging.info("cuda availableness check start.")
    import chainer
    import torch
    try:
        assert torch.cuda.is_available()
        logging.info("--> cuda is available in torch.")
    except AssertionError:
        logging.warn("--> it seems that cuda is not available in torch.")
    try:
        assert torch.backends.cudnn.is_available()
        logging.info("--> cudnn is available in torch.")
    except AssertionError:
        logging.warn("--> it seems that cudnn is not available in torch.")
    try:
        assert chainer.backends.cuda.available
        logging.info("--> cuda is available in chainer.")
    except AssertionError:
        logging.warn("--> it seems that cuda is not available in chainer.")
    try:
        assert chainer.backends.cuda.cudnn_enabled
        logging.info("--> cudnn is available in chainer.")
    except AssertionError:
        logging.warn("--> it seems that cudnn is not available in chainer.")
    try:
        from cupy.cuda import nccl  # NOQA
        logging.info("--> nccl is installed.")
    except ImportError:
        logging.warn("--> it seems that nccl is not installed. multi-gpu is not enabled.")
        logging.warn("--> if you want to use multi-gpu, please install it and then re-setup.")
    try:
        assert torch.cuda.device_count() > 1
        logging.info("--> multi-gpu is available (#gpus = %d)." % torch.cuda.device_count())
    except AssertionError:
        logging.warn("--> it seems that only single gpu is available.")
        logging.warn('--> maybe your machine has only one gpu.')
    logging.info("cuda availableness check done.")
