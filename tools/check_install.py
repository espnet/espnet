#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s")

logging.info("python version = " + sys.version)

# check library availableness
try:
    import chainer
    logging.info("chainer is installed.")
except ModuleNotFoundError:
    logging.warn("chainer is not installed.")
try:
    import cupy  # NOQA
    logging.info("cupy is installed.")
except ModuleNotFoundError:
    logging.warn("cupy is not installed.")
try:
    import inflect  # NOQA
    logging.info("inflect is installed.")
except ModuleNotFoundError:
    logging.warn("inflect is not installed.")
try:
    import librosa  # NOQA
    logging.info("librosa is installed.")
except ModuleNotFoundError:
    logging.warn("librosa is not installed.")
try:
    import matplotlib  # NOQA
    logging.info("matplotlib is installed.")
except ModuleNotFoundError:
    logging.warn("matplotlib is not installed.")
try:
    import soundfile  # NOQA
    logging.info("soundfile is installed.")
except ModuleNotFoundError:
    logging.warn("soundfile is not installed.")
try:
    import torch  # NOQA
    logging.info("torch is installed.")
except ModuleNotFoundError:
    logging.warn("torch is not installed.")
try:
    import unidecode  # NOQA
    logging.info("unidecode is installed.")
except ModuleNotFoundError:
    logging.warn("unidecode is not installed.")
try:
    import warpctc_pytorch  # NOQA
    logging.info("pytorch warpctc is installed.")
except ModuleNotFoundError:
    logging.warn("pytorch warpctc is not installed.")
try:
    import chainer_ctc  # NOQA
    logging.info("chainer warpctc is installed")
except ModuleNotFoundError:
    logging.warn("chainer warpctc is not installed")

# check version
if 'torch' in globals():
    try:
        assert torch.__version__ == '0.4.1'
        logging.info("torch version is matched.")
    except AssertionError:
        logging.warn("torch version is not matched. please install torch==0.4.1.")
if 'chainer' in globals():
    try:
        assert chainer.__version__ == '4.3.1'
        logging.info("chainer version is matched.")
    except AssertionError:
        logging.warn("chainer version is not matched. please install chainer==4.3.1.")

# check cuda availableness
if 'torch' in globals():
    try:
        assert torch.cuda.is_available()
        logging.info("cuda is available in torch.")
    except AssertionError:
        logging.warn("it seems that cuda is not available in torch.")
    try:
        assert torch.backends.cudnn.is_available()
        logging.info("cudnn is available in torch.")
    except AssertionError:
        logging.warn("it seems that cudnn is not available in torch.")
if 'chainer' in globals():
    try:
        assert chainer.backends.cuda.available
        logging.info("cuda is available in chainer.")
    except AssertionError:
        logging.warn("it seems that cuda is not available in chainer.")
    try:
        assert chainer.backends.cuda.cudnn_enabled
        logging.info("cudnn is available in chainer.")
    except AssertionError:
        logging.warn("it seems that cudnn is not available in chainer.")
if 'cupy' in globals():
    try:
        from cupy.cuda import nccl  # NOQA
        logging.info("nccl is installed.")
    except ImportError:
        logging.warn("nccl is not installed. multi-gpu is not enabled.")
if 'torch' in globals() and torch.cuda.is_available():
    try:
        assert torch.cuda.device_count() > 1
        logging.info("multi-gpu is available (#gpus = %d)." % torch.cuda.device_count())
    except AssertionError:
        logging.warn("it seems that only single gpu is available.")
