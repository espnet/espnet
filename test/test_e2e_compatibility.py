#!/usr/bin/env python
# coding: utf-8

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import os
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.pytorch_backend.e2e_asr import E2E


IS_PY3 = platform.python_version_tuple()[0] == '3'


def download_zip_from_google_drive(download_dir, file_id):
    os.makedirs(download_dir, exist_ok=True)

    # download zip file from google drive
    cmd = ["wget", "https://drive.google.com/uc?export=download&id=%s" % file_id, "-O", download_dir + "/tmp.zip"]
    cmd_state = subprocess.run(cmd, check=True)

    # check
    if cmd_state.returncode != 0:
        print("download failed.")
        sys.exit(cmd_state.returncode)

    # unzip downloaded files
    cmd = ["unzip", download_dir + "/tmp.zip", "-d", download_dir]
    cmd_state = subprocess.run(cmd, check=True)

    # check
    if cmd_state.returncode != 0:
        print("unzip failed.")
        sys.exit(cmd_state.returncode)

    # get model file path
    cmd = ["find", download_dir, "-name", "model.acc.best"]
    cmd_state = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)

    # check
    if cmd_state.returncode != 0:
        print("find failed.")
        sys.exit(cmd_state.returncode)

    return cmd_state.stdout.decode("utf-8").split("\n")[0]


@pytest.mark.skipif(not IS_PY3, reason="not support python 2")
@pytest.mark.parametrize("file_id", ["1zF88bRNbJhw9hNBq3NrDg8vnGGibREmg"])
def test_pytorch_downloaded_model_decodable(file_id):
    # download model
    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir=".")
    model_path = download_zip_from_google_drive(tmpdir, file_id)

    # load trained model parameters
    idim, odim, train_args = get_model_conf(model_path)
    model = E2E(idim, odim, train_args)
    torch_load(model_path, model)

    with torch.no_grad():
        in_data = np.random.randn(128, idim)
        model.recognize(in_data, train_args, train_args.char_list)  # decodable

    # remove
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
