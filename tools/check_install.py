#!/usr/bin/env python3

"""Script to check whether the installation is done correctly."""

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.version import parse

module_list = [
    ("torchaudio", None, None),
    ("torch_optimizer", None, None),
    ("warprnnt_pytorch", None, "installers/install_warp-transducer.sh"),
    ("chainer_ctc", None, "installers/install_chainer_ctc.sh"),
    ("pyopenjtalk", None, "installers/install_pyopenjtalk.sh"),
    ("tdmelodic_pyopenjtalk", None, "installers/install_tdmelodic_pyopenjtalk.sh"),
    ("kenlm", None, "installers/install_kenlm.sh"),
    ("mmseg", None, "installers/install_py3mmseg.sh"),
    ("espnet", None, None),
    ("numpy", None, None),
    ("fairseq", None, "installers/install_fairseq.sh"),
    ("phonemizer", None, "installers/install_phonemizer.sh"),
    ("gtn", None, "installers/install_gtn.sh"),
    ("s3prl", None, "installers/install_s3prl.sh"),
    ("transformers", None, "installers/install_transformers.sh"),
    ("speechbrain", None, "installers/install_speechbrain.sh"),
    ("k2", None, "installers/install_k2.sh"),
    ("longformer", None, "installers/install_longformer.sh"),
    ("nlg-eval", None, "installers/install_longformer.sh"),
    ("datasets", None, "installers/install_longformer.sh"),
    ("pykeops", None, "installers/install_cauchy_mult.sh"),
    ("whisper", None, "installers/install_whisper.sh"),
    ("RawNet3", None, "installers/install_rawnet.sh"),
    ("reazonspeech", None, "installers/install_reazonspeech.sh"),
]

executable_list = [
    ("sclite", "installers/install_sctk.sh", None),
    ("sph2pipe", "installers/install_sph2pipe.sh", None),
    ("PESQ", "installers/install_pesq.sh", None),
    ("BeamformIt", "installers/install_beamformit.sh", None),
    ("spm_train", None, None),
    ("spm_encode", None, None),
    ("spm_decode", None, None),
    ("sox", None, "--version"),
    ("ffmpeg", None, "-version"),
    ("flac", None, "--version"),
    ("cmake", None, "--version"),
]


def main():
    """Check the installation."""

    python_version = sys.version.replace("\n", " ")
    print(f"[x] python={python_version}")

    print()
    print("Python modules:")
    try:
        import torch

        print(f"[x] torch={torch.__version__}")

        if torch.cuda.is_available():
            print(f"[x] torch cuda={torch.version.cuda}")
        else:
            print("[ ] torch cuda")

        if torch.backends.cudnn.is_available():
            print(f"[x] torch cudnn={torch.backends.cudnn.version()}")
        else:
            print("[ ] torch cudnn")

        if torch.distributed.is_nccl_available():
            print("[x] torch nccl")
        else:
            print("[ ] torch nccl")

    except ImportError:
        print("[ ] torch")

    try:
        import chainer

        print(f"[x] chainer={chainer.__version__}")
        if parse(chainer.__version__) != parse("6.0.0"):
            print(
                f"Warning! chainer={chainer.__version__} is not supported. "
                "Supported version is 6.0.0"
            )

        if chainer.backends.cuda.available:
            print("[x] chainer cuda")
        else:
            print("[ ] chainer cuda")

        if chainer.backends.cuda.cudnn_enabled:
            print("[x] chainer cudnn")
        else:
            print("[ ] chainer cudnn")

    except ImportError:
        print("[ ] chainer")

    try:
        import cupy

        print(f"[x] cupy={cupy.__version__}")
        try:
            from cupy.cuda import nccl  # NOQA

            print("[x] cupy nccl")
        except ImportError:
            print("[ ] cupy nccl")
    except ImportError:
        print("[ ] cupy")

    to_install = []
    for name, versions, installer in module_list:
        try:
            m = importlib.import_module(name)
            if hasattr(m, "__version__"):
                version = m.__version__
                print(f"[x] {name}={version}")
                if versions is not None and version not in versions:
                    print(
                        f"Warning! {name}={version} is not suppoted. "
                        "Supported versions are {versions}"
                    )
            else:
                print(f"[x] {name}")
        except ImportError:
            print(f"[ ] {name}")
            if installer is not None:
                to_install.append(f"Use '{installer}' to install {name}")

    # check muskits install
    if Path("muskits.done").exists():
        print("[x] muskits")
    else:
        print("[ ] muskits")
        to_install.append("Use 'installers/install_muskits.sh' to install muskits")

    if not Path("kaldi/egs/wsj/s5/utils/parse_options.sh").exists():
        print("[ ] Kaldi")
        to_install.append(
            "Type 'git clone --depth 1 https://github.com/kaldi-asr/kaldi'"
            " and see 'kaldi/tools/INSTALL' to install Kaldi"
        )
    elif not Path("kaldi/src/bin/copy-matrix").exists():
        print("[x] Kaldi (not compiled)")
        to_install.append("See 'kaldi/tools/INSTALL' to install Kaldi")
    else:
        print("[x] Kaldi (compiled)")

    print()
    print("Executables:")

    pattern = re.compile(r"([0-9]+.[0-9]+.[0-9]+[^\s]*)\s*")

    for name, installer, version_option in executable_list:
        if shutil.which(name) is not None:
            string = f"[x] {name}"
            if version_option is not None:
                cp = subprocess.run(
                    [name, version_option], capture_output=True, text=True
                )
                if cp.returncode == 0:
                    ma = re.search(pattern, cp.stdout)
                    if ma is not None:
                        string = f"[x] {name}={ma.group(1)}"
                    else:
                        ma = re.search(pattern, cp.stderr)
                        if ma is not None:
                            string = f"[x] {name}={ma.group(1)}"
            print(string)

        else:
            print(f"[ ] {name}")
            if installer is not None:
                to_install.append(f"Use '{installer}' to install {name}")

    print()
    print("INFO:")
    for m in to_install:
        print(m)


if __name__ == "__main__":
    main()
