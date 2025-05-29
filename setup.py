#!/usr/bin/env python3

"""ESPnet setup script."""

import os

from setuptools import find_packages, setup

requirements = {
    "install": [
        "typeguard",
        "torch",
        "torchaudio",
        "numpy",
        "espnet_tts_frontend",
        "asteroid_filterbanks",
        "transformers",
        "torchaudio",
        "einops",
        "humanfriendly",
        "torch-complex",
        "h5py",
        "kaldiio",
        "soundfile",
        "librosa",
        "sentencepiece",
    ],
}

install_requires = requirements["install"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "espnet", "version.txt")
with open(version_file, "r") as f:
    version = f.read().strip()
setup(
    name="espnet",
    version=version,
    url="http://github.com/espnet/espnet",
    author="Shinji Watanabe",
    author_email="shinjiw@ieee.org",
    description="ESPnet: end-to-end speech processing toolkit",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache Software License",
    packages=find_packages(include=["espnet*"]),
    package_data={"espnet": ["version.txt"]},
    # #448: "scripts" is inconvenient for developping because they are copied
    # scripts=get_all_scripts('espnet/bin'),
    install_requires=install_requires,
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
