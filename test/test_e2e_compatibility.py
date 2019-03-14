#!/usr/bin/env python
# coding: utf-8

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import importlib
import os
import platform
import shutil
import subprocess
import tempfile

import chainer
import numpy as np
import pytest
import torch

from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load


IS_PY3 = platform.python_version_tuple()[0] == '3'


def download_zip_from_google_drive(download_dir, file_id):
    # directory check
    os.makedirs(download_dir, exist_ok=True)

    # download zip file from google drive via wget
    cmd = ["wget", "https://drive.google.com/uc?export=download&id=%s" % file_id, "-O", download_dir + "/tmp.zip"]
    subprocess.run(cmd, check=True)

    try:
        # unzip downloaded files
        cmd = ["unzip", download_dir + "/tmp.zip", "-d", download_dir]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # sometimes, wget is failed due to vrius check in google drive
        # to avoid it, we need to do some tricky processing
        # see https://stackoverflow.com/questions/20665881/direct-download-from-google-drive-using-google-drive-api
        subprocess.call("curl -c /tmp/cookies "
                        "\"https://drive.google.com/uc?export=download&id=%s\" "
                        "> /tmp/intermezzo.html" % file_id, shell=True)
        subprocess.call("curl -L -b /tmp/cookies \"https://drive.google.com$(cat /tmp/intermezzo.html "
                        "| grep -Po \'uc-download-link\" [^>]* href=\"\K[^\"]*\' "  # NOQA
                        "| sed \'s/\&amp;/\&/g\')\" > %s" % (download_dir + "/tmp.zip"), shell=True)  # NOQA
        cmd = ["unzip", download_dir + "/tmp.zip", "-d", download_dir]
        subprocess.run(cmd, check=True)

    # get model file path
    cmd = ["find", download_dir, "-name", "model.*.best"]
    cmd_state = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)

    return cmd_state.stdout.decode("utf-8").split("\n")[0]


# TODO(kan-bayashi): make it to be compatible with python2
# file id in google drive can be obtain from sharing link
# ref: https://qiita.com/namakemono/items/c963e75e0af3f7eed732
@pytest.mark.skipif(not IS_PY3, reason="not support python 2")
@pytest.mark.parametrize("module, download_info", [
    ("espnet.nets.pytorch_backend.e2e_asr", ("v.0.3.0 egs/an4/asr1 pytorch", "1zF88bRNbJhw9hNBq3NrDg8vnGGibREmg")),
    ("espnet.nets.chainer_backend.e2e_asr", ("v.0.3.0 egs/an4/asr1 chainer", "1m2SZLNxvur3q13T6Zrx6rEVfqEifgPsx"))
])
def test_downloaded_asr_model_decodable(module, download_info):
    # download model
    print(download_info[0])
    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir=".")
    model_path = download_zip_from_google_drive(tmpdir, download_info[1])

    # load trained model parameters
    m = importlib.import_module(module)
    idim, odim, train_args = get_model_conf(model_path)
    model = m.E2E(idim, odim, train_args)
    if "chainer" in module:
        chainer_load(model_path, model)
    else:
        torch_load(model_path, model)

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(128, idim)
        model.recognize(in_data, train_args, train_args.char_list)  # decodable

    # remove
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
