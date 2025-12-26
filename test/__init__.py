"""Test package for Hydra instantiation targets.

Some espnet tests use Hydra `_target_` dotted paths pointing into `test.*`.
Making `test/` a proper Python package ensures these targets resolve to the
local test modules (instead of the stdlib `test` package).
"""
