"""Utilities for building machine learning workflows.

This package aggregates the core components used to build and train machine-learning
models.  It re-exports the most frequently used classes, functions and configuration
helpers so that they can be imported directly from the package root.

Available modules
-----------------
.. autosummary::
   :toctree: _autosummary

   .config
   .data
   .dataset
   .preprocess
   .task
   .trainer

Public API
----------
The following symbols are exported from the package namespace:

* :class:`.trainer.Trainer` - orchestrates training loops and evaluation.
* Everything imported from :mod:`.config`, :mod:`.data`, :mod:`.dataset`,
  :mod:`.preprocess` and :mod:`.task` via ``import *``.
  These include configuration helpers, data loaders, dataset abstractions,
  preprocessing utilities and task definitions.

Typical usage
-------------
>>> from mypackage import Trainer, Task, Dataset, preprocess, load_dataset
>>> trainer = Trainer()
>>> trainer.train(model, dataset, task)

This design allows developers to quickly assemble training pipelines without
having to import each submodule individually.  All public symbols are documented
in their respective modules; consult the module docs for detailed usage.
"""

from .config import *  # noqa
from .data import *  # noqa
from .dataset import *  # noqa
from .preprocess import *  # noqa
from .task import *  # noqa
from .trainer import Trainer  # noqa
