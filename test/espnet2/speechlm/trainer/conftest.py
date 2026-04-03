"""Dependency stubs for espnet2/speechlm/trainer tests.

Injects lightweight stubs for heavy optional dependencies (deepspeed,
wandb, humanfriendly) so that tests can run in CI without those packages.
"""

import importlib.machinery
import sys
import types

import torch.nn as nn


def _install_stub(name, module):
    sys.modules.setdefault(name, module)


def pytest_configure():
    """Inject stubs before any test module is collected."""

    # ---- deepspeed stub ----
    if "deepspeed" not in sys.modules:
        ds = types.ModuleType("deepspeed")
        ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)

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

    # ---- wandb stub ----
    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)

        class _MockRun:
            pass

        class _MockConfig:
            def update(self, d):
                pass

        wandb.run = _MockRun()
        wandb.config = _MockConfig()

        def log(data, step=None):
            pass

        wandb.log = log
        _install_stub("wandb", wandb)

    # ---- humanfriendly stub ----
    if "humanfriendly" not in sys.modules:
        humanfriendly = types.ModuleType("humanfriendly")
        humanfriendly.__spec__ = importlib.machinery.ModuleSpec(
            "humanfriendly", loader=None
        )

        def format_size(num_bytes, **kwargs):
            return f"{num_bytes} bytes"

        humanfriendly.format_size = format_size
        _install_stub("humanfriendly", humanfriendly)
