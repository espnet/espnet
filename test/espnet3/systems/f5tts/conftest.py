"""Shared setup for the F5-TTS tests.

CI runs ``pytest --execution-timeout 10.0``, a per-test budget that covers the
call phase only. The pinyin tokenizer's first use pays a one-off cost (loading
rjieba's Rust dictionary and pypinyin's tables) which on the slower Python 3.12
runners pushes the first test that touches it past that budget. When the
timeout fires mid-import the module never lands in ``sys.modules``, so every
following test repeats the same work and fails the same way.

Paying it here moves the cost into fixture setup, which the execution timeout
does not cover, and leaves every test to hit a warm import.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def warm_optional_tokenizer_dependencies():
    """Import and initialise rjieba and pypinyin once, outside any test."""
    try:
        import rjieba
        from pypinyin import lazy_pinyin
    except ImportError:
        # Both are optional extras; the tests that need them skip themselves.
        return

    # Touch each one so the dictionaries load now rather than inside a test.
    rjieba.cut("你好")
    lazy_pinyin("你好")
