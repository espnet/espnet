"""Collection gate for espnet2/speechlm/model tests.

``espnet2/speechlm/model/__init__.py`` eagerly imports
``SpeechLMJobTemplate`` → ``lm/parallel.py`` → ``lm/loss.py``, and
``lm/loss.py`` imports ``liger_kernel`` at module-load time.
``liger_kernel`` is listed in the ``espnet[speechlm]`` extra but is
optional in the base install. When it (or its fused-CE submodule) is
not importable, the whole model subtree would fail at collection; we
``collect_ignore_glob`` it instead so the rest of the test suite keeps
running.

No other stubs here. Heavy deps (``transformers``, ``joblib``,
``humanfriendly``) are real packages in CI, so we use them directly.
Per-test monkeypatching of, e.g., ``transformers.AutoConfig`` lives in
the test files themselves so that a global patch can never leak into
unrelated tests.
"""


def _liger_fused_ce_importable():
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (  # noqa: F401
            LigerFusedLinearCrossEntropyFunction,
        )

        return True
    except Exception:
        return False


if not _liger_fused_ce_importable():
    # Skip every .py under this conftest's directory, including the ones
    # alongside conftest itself and any depth of nested subfolders. pytest
    # only walks two separate match passes (same-dir / nested), so we list
    # both patterns explicitly.
    collect_ignore_glob = ["*.py", "*/*.py", "*/*/*.py"]
