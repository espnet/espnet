"""Dependency shims for espnet2/speechlm/trainer tests.

Where possible we import the real package. Two categories need shims:

1. **wandb** — real wandb is installable, but `wandb.init()` contacts a
   remote service or local state that isn't present in CI. We stub it so
   tests don't need credentials or network.

2. **deepspeed** — not part of ``espnet[speechlm]`` (and not part of base
   deps). DeepSpeed is GPU-oriented and heavy, so CI doesn't install it.

3. **``espnet2.speechlm.model.speechlm.parallel_utils``** — this package
   is being added in a separate PR and isn't present on the PR's base
   branch. A stub keeps imports working until that PR lands.

Everything else (humanfriendly, torchtitan, transformers, ...) is
imported as the real package so version mismatches are caught by tests.
"""

import importlib.machinery
import sys
import types

import torch.nn as nn


def _install_stub(name, module):
    sys.modules.setdefault(name, module)


def _make_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


def pytest_configure():
    """Inject shims that are genuinely needed for CPU-only CI."""

    # ---- deepspeed shim ----
    # Not in `espnet[speechlm]` and not a base dep. Tests of the
    # DeepSpeedTrainer pathway mock the engine interface below.
    if "deepspeed" not in sys.modules:
        ds = _make_module("deepspeed")

        def initialize(model=None, model_parameters=None, config=None, **kwargs):
            class _MockEngine(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.module = model

                def forward(self, **kwargs):
                    return self.module(**kwargs)

                def train(self, mode=True):
                    self.module.train(mode)
                    return self

                def eval(self):
                    self.module.eval()
                    return self

                def backward(self, loss):
                    loss.backward()

                def step(self):
                    pass

                def get_lr(self):
                    return [1e-4]

                def get_global_grad_norm(self):
                    return 0.0

                def save_checkpoint(self, path, client_state=None):
                    pass

                def load_checkpoint(self, path):
                    return None, None

                def parameters(self):
                    return self.module.parameters()

                def named_parameters(self):
                    return self.module.named_parameters()

            engine = _MockEngine(model)
            return engine, None, None, None

        ds.initialize = initialize
        _install_stub("deepspeed", ds)

    # ---- wandb shim ----
    # Real wandb is installed via ``espnet[speechlm]``, but its APIs
    # (config.update, log) refuse to run until wandb.init() has been
    # called, and wandb.init without credentials would either block on
    # login or emit real run metadata we don't want in CI. Force-override
    # sys.modules so test code always sees the no-op stub — this runs
    # before any test module imports the trainer, so the replacement
    # sticks.
    wandb = _make_module("wandb")

    class _MockRun:
        pass

    class _MockConfig:
        def update(self, d):
            pass

    wandb.run = _MockRun()
    wandb.config = _MockConfig()
    wandb.log = lambda data, step=None: None
    wandb.init = lambda *a, **k: _MockRun()
    wandb.finish = lambda *a, **k: None
    sys.modules["wandb"] = wandb

    # ---- torchtitan fallback shim ----
    # ``torchtitan`` is listed in ``espnet[speechlm]`` and CI installs it
    # as a real package. The fallback shim only activates in environments
    # where the user opted out of the speechlm extra — so version errors
    # in real torchtitan still surface in CI.
    if "torchtitan" not in sys.modules:
        try:
            import torchtitan  # noqa: F401
            from torchtitan.distributed import utils as _tt_utils  # noqa: F401
        except Exception:
            tt = _make_module("torchtitan")
            tt_dist = _make_module("torchtitan.distributed")
            tt_dist_utils = _make_module("torchtitan.distributed.utils")

            import torch as _torch

            def _stub_clip_grad_norm_(parameters, max_norm, **kwargs):
                return _torch.tensor(0.0)

            tt_dist_utils.clip_grad_norm_ = _stub_clip_grad_norm_
            tt_dist.utils = tt_dist_utils
            tt.distributed = tt_dist
            _install_stub("torchtitan", tt)
            _install_stub("torchtitan.distributed", tt_dist)
            _install_stub("torchtitan.distributed.utils", tt_dist_utils)

    # ---- parallel_utils shim ----
    # ``espnet2.speechlm.model.speechlm.parallel_utils`` is being added in
    # a separate PR. When it's absent or is merely a namespace-package
    # placeholder (leftover pyc dir without __init__.py), install a
    # lightweight stub so TitanTrainer can still be imported.
    pu_name = "espnet2.speechlm.model.speechlm.parallel_utils"
    _real_pu = None
    try:
        __import__(pu_name)
        _real_pu = sys.modules.get(pu_name)
    except Exception:
        _real_pu = None

    if _real_pu is None or not hasattr(_real_pu, "init_parallel_dims"):
        pu = _make_module(pu_name)

        class _StubMesh:
            def __init__(self, size=1):
                self._size = size

            def size(self):
                return self._size

            def get_group(self):
                return None

            def get_local_rank(self):
                return 0

            def __getitem__(self, key):
                return _StubMesh(self._size)

        class _StubParallelDims:
            def __init__(self):
                self.dp_replicate = 1
                self.dp_shard = 1
                self.tp = 1
                self.pp = 1
                self.ep = 1
                self.world_size = 1
                self.fsdp_enabled = False
                self.dp_replicate_enabled = False
                self.pp_enabled = False
                self.ep_enabled = False

            def get_mesh(self, name):
                return _StubMesh(1)

            def get_optional_mesh(self, name):
                return None

        def _stub_init_parallel_dims(titan_config):
            return _StubParallelDims(), 0, 0

        def _stub_parallelize(model, parallel_dims, titan_config, **kwargs):
            return model

        def _stub_build_pipeline(model, parallel_dims, titan_config, **kwargs):
            class _Schedule:
                def step(self, *args, **kwargs):
                    return None

            return _Schedule(), True

        pu.init_parallel_dims = _stub_init_parallel_dims
        pu.parallel_strategies = {"qwen3": _stub_parallelize}
        pu.build_pipeline = _stub_build_pipeline
        pu._StubMesh = _StubMesh
        pu._StubParallelDims = _StubParallelDims
        sys.modules[pu_name] = pu
