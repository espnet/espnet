#!/usr/bin/env python3

"""Script to check whether the installation is done correctly."""

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import importlib
import shutil
import sys

from packaging.version import parse

module_list = [
    ("torchaudio", None, None),
    ("torch_optimizer", None, None),
    ("warpctc_pytorch", None, "installers/install_warp-ctc.sh"),
    ("warprnnt_pytorch", None, "installers/install_warp-transducer.sh"),
    ("chainer_ctc", None, "installers/install_chainer_ctc.sh"),
    ("pyopenjtalk", None, "installers/install_pyopenjtalk.sh"),
    ("tdmelodic_pyopenjtalk", None, "installers/install_tdmelodic_pyopenjtalk.sh"),
    ("kenlm", None, "installers/install_kenlm.sh"),
    ("mmseg", None, "installers/install_py3mmseg.sh"),
    ("espnet", None, None),
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
]

executable_list = [
    ("sclite", "installers/install_sctk.sh"),
    ("sph2pipe", "installers/install_sph2pipe.sh"),
    ("PESQ", "installers/install_pesq.sh"),
    ("BeamformIt", "installers/install_beamformit.sh"),
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

    print()
    print("Executables:")
    for name, installer in executable_list:
        if shutil.which(name) is not None:
            print(f"[x] {name}")
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
