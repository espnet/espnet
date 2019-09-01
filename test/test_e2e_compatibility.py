#!/usr/bin/env python3
# coding: utf-8

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import importlib
import os
from os.path import join
import re
import shutil
import six
import subprocess
import tempfile

import chainer
import numpy as np
import pytest
import torch

from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load


def download_zip_from_google_drive(download_dir, file_id):
    # directory check
    os.makedirs(download_dir, exist_ok=True)
    tmpzip = join(download_dir, "tmp.zip")

    # download zip file from google drive via wget
    cmd = ["wget", "https://drive.google.com/uc?export=download&id=%s" % file_id, "-O", tmpzip]
    subprocess.run(cmd, check=True)

    try:
        # unzip downloaded files
        cmd = ["unzip", tmpzip, "-d", download_dir]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # sometimes, wget from google drive is failed due to virus check confirmation
        # to avoid it, we need to do some tricky processings
        # see https://stackoverflow.com/questions/20665881/direct-download-from-google-drive-using-google-drive-api
        out = subprocess.check_output("curl -c /tmp/cookies "
                                      "\"https://drive.google.com/uc?export=download&id=%s\""
                                      % file_id, shell=True)
        out = out.decode("utf-8")
        dllink = "https://drive.google.com{}".format(re.findall(
            r'<a id="uc-download-link" [^>]* href="([^"]*)">', out)[0].replace('&amp;', '&'))
        subprocess.call(f"curl -L -b /tmp/cookies \"{dllink}\" > {tmpzip}", shell=True)  # NOQA
        cmd = ["unzip", tmpzip, "-d", download_dir]
        subprocess.run(cmd, check=True)

    # get model file path
    cmd = ["find", download_dir, "-name", "model.*.best"]
    cmd_state = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)

    return cmd_state.stdout.decode("utf-8").split("\n")[0]


# TODO(kan-bayashi): make it to be compatible with python2
# file id in google drive can be obtain from sharing link
# ref: https://qiita.com/namakemono/items/c963e75e0af3f7eed732
@pytest.mark.skipif(not six.PY3, reason="not support python 2")
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
