# To use a consistent encoding
from codecs import open
from os import path

import numpy
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# First check if cupy is installed, we
try:
    import cupy
except ImportError:
    raise RuntimeError(
        "CuPy is not available. Please install it manually: "
        "https://docs.cupy.dev/en/stable/install.html"
    )

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chime7task1",
    version="0.0.1",
    description="CHiME-7 Task 1 Official Baseline Recipe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/desh2608/gss",
    author="Desh Raj",
    author_email="r.desh26@gmail.com",
    keywords="speech enhancement gss",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    install_requires=[
        "cached_property",
        "numpy",
        "lhotse @ git+http://github.com/lhotse-speech/lhotse",
    ],
    extras_require={
        "dev": dev_requires,
    },
    include_dirs=[numpy.get_include()],
    entry_points={
        "console_scripts": [
            "gss=gss.bin.gss:cli",
        ]
    },
)