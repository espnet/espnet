"""
Enables multiple commonly used features.

Method resolution order:

- `tqdm.autonotebook` without import warnings
- `tqdm.asyncio` on Python3.6+
- `tqdm.std` base class

Usage:
>>> from tqdm.auto import trange, tqdm
>>> for i in trange(10):
...     ...
"""
from .std import tqdm, trange

__all__ = ["tqdm", "trange"]
