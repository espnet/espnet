"""Initialize espnet2 package and set __version__."""

import os
import warnings
from importlib import metadata as _metadata

_here = os.path.dirname(__file__)
_top_version_file = os.path.join(os.path.dirname(_here), "version.txt")
_pkg_version_file = os.path.join(_here, "version.txt")

__version__ = "0.0.0"

try:
    __version__ = _metadata.version("espnet")
except Exception:
    if os.path.exists(_top_version_file):
        with open(_top_version_file, "r") as f:
            __version__ = f.read().strip()
    elif os.path.exists(_pkg_version_file):
        with open(_pkg_version_file, "r") as f:
            __version__ = f.read().strip()


def _warn_if_torch_torchaudio_skewed() -> None:
    """Explain the OSError a mismatched torch/torchaudio pair raises.

    torchaudio ships a compiled extension linked against torch's ABI, so a
    version skew fails at ``import torchaudio`` with

        OSError: Could not load this library: .../torchaudio/lib/_torchaudio.abi3.so

    which names neither package and says nothing about versions. pip can arrive
    at such a pair on its own: torchaudio publishes no dependency metadata, so
    nothing links a release to the torch it was built against.

    The versions are read from distribution metadata rather than by importing
    either package - so this still works in exactly the case it exists to
    explain. It adds no import; two metadata lookups cost about 0.7 ms.
    """
    try:
        torch_version = _metadata.version("torch")
        torchaudio_version = _metadata.version("torchaudio")
    except _metadata.PackageNotFoundError:
        # torchaudio is optional for some code paths, and an absent torch is
        # not something this can usefully say anything about.
        return

    # Compare major.minor only: the local version suffix ("2.11.0+cu126") is
    # not part of the pairing.
    torch_minor = torch_version.split(".")[:2]
    torchaudio_minor = torchaudio_version.split(".")[:2]
    if torch_minor == torchaudio_minor:
        return

    # torchaudio tracked torch's minor exactly up to 2.11.0, which is its last
    # release and the first built against PyTorch's stable ABI - its libraries
    # import only the aoti_torch_* C API, no mangled C++ torch symbols, so one
    # build keeps working as torch moves on. It is therefore the matched pair
    # for every torch from 2.11 on, and the pair install_torch.sh installs
    # above 2.11.0, there being no later torchaudio to install. Checked on
    # torch 2.13.0 and 2.14.0: resample, MelSpectrogram, Spectrogram and
    # compliance.kaldi.fbank all work.
    try:
        torch_pair = tuple(int(part) for part in torch_minor)
    except ValueError:
        # A version this cannot parse is not one to reassure anyone about.
        torch_pair = ()
    if torchaudio_minor == ["2", "11"] and torch_pair >= (2, 11):
        return

    warnings.warn(
        f"torch {torch_version} is installed alongside torchaudio "
        f"{torchaudio_version}. torchaudio's compiled extension is built "
        "against a specific torch ABI, so importing it may fail with "
        '"OSError: Could not load this library: ..._torchaudio.abi3.so", '
        "which does not mention either version. Install the two as a matched "
        "pair - tools/installers/install_torch.sh does.",
        RuntimeWarning,
        stacklevel=2,
    )


_warn_if_torch_torchaudio_skewed()
