"""Tests for espnet3.utils.stage_cli: wiring CLI flags from StageSpec."""

import argparse

import pytest

from espnet3.contract.stages import StageContractError, StageSpec
from espnet3.utils.stage_cli import (
    add_stage_flags,
    build_stage_flag_parser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_stage_dests(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    """Return the set of all registered destination keys for stage flags."""
    return tuple(
        action.dest
        for action in parser._actions
        if hasattr(action, "dest") and action.dest not in ("help",)
    )


# ---------------------------------------------------------------------------
# add_stage_flags: basic behavior
# ---------------------------------------------------------------------------


def test_add_stage_flags_creates_exp_dir_flag():
    """Basic: log_dir='exp_dir' -> --exp-dir writes to dest 'exp_dir'."""
    spec = StageSpec(name="train", config="training", log_dir="exp_dir")
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec,))

    dests = _extract_stage_dests(parser)
    assert "exp_dir" in dests

    args = parser.parse_args(["--exp-dir", "/tmp/exp"])
    assert args.exp_dir == "/tmp/exp"


def test_add_stage_flags_creates_dir_with_dots():
    """Map dotted OmegaConf keys to underscore CLI destinations."""
    spec = StageSpec(name="tokenize", config="tokenizer", log_dir="tokenizer.save_path")
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec,))

    dests = _extract_stage_dests(parser)
    # Destination is safe identifier version of config key
    assert "tokenizer_save_path" in dests

    args = parser.parse_args(["--tokenizer-save-path", "/tmp/save"])
    assert args.tokenizer_save_path == "/tmp/save"


def test_add_stage_flags_with_alias_collects_all_names_same_dest():
    """Alias + canonical -> both flags write to SAME canonical destination."""
    spec = StageSpec(
        name="train",
        config="training",
        log_dir="exp_dir",
        dir_aliases=("training_dir",),
    )
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec,))

    # Both CLI flags exist
    args1 = parser.parse_args(["--exp-dir", "/a"])
    args2 = parser.parse_args(["--training-dir", "/a"])
    assert args1.exp_dir == "/a"
    assert args2.exp_dir == "/a"  # Same dest key, both set it
    # Verify that destination is canonical
    dests = _extract_stage_dests(parser)
    assert "exp_dir" in dests


def test_add_stage_flags_from_two_stages_with_same_canonical():
    """Two stages sharing canonical key create one flag."""
    spec1 = StageSpec(name="train", config="training", log_dir="output")
    spec2 = StageSpec(name="infer", config="inference", log_dir="output")
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec1, spec2))

    args = parser.parse_args(["--output", "/shared"])
    assert args.output == "/shared"


def test_add_stage_flags_with_aliases_from_shared_key_all_work():
    """Shared key where both stages have aliases: all 3 flags must write to

    the SAME destination - the declared canonical key "output" - not an
    arbitrary alphabetically-first alias like "out_dir".
    """
    spec1 = StageSpec(
        name="train", config="training", log_dir="output", dir_aliases=("out_dir",)
    )
    spec2 = StageSpec(
        name="infer", config="inference", log_dir="output", dir_aliases=("results",)
    )
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec1, spec2))

    assert parser.parse_args(["--output", "/base"]).output == "/base"
    assert parser.parse_args(["--out-dir", "/alt1"]).output == "/alt1"
    assert parser.parse_args(["--results", "/alt2"]).output == "/alt2"

    dests = _extract_stage_dests(parser)
    assert set(dests) == {"output"}


def test_add_stage_flags_handles_no_log_dir():
    """log_dir=None produces no stage flags."""
    spec = StageSpec(name="foo", config="bar", log_dir=None)
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec,))

    dests = _extract_stage_dests(parser)
    assert len(dests) == 0


# ---------------------------------------------------------------------------
# add_stage_flags: prefix behavior
# ---------------------------------------------------------------------------


def test_add_stage_flags_with_prefix_prepends():
    """Prefix adds prefix_ to CLI flag names."""
    spec = StageSpec(name="train", config="training", log_dir="exp_dir")
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec,), prefix="stats")

    dests = _extract_stage_dests(parser)
    assert "exp_dir" in dests

    args = parser.parse_args(["--stats-exp-dir", "/a"])
    assert args.exp_dir == "/a"


# ---------------------------------------------------------------------------
# build_stage_flag_parser: convenience wrapper
# ---------------------------------------------------------------------------


def test_build_stage_flag_parser_returns_new_parser():
    """Convenience function builds a new parser with stage flags."""
    spec = StageSpec(name="train", config="training", log_dir="exp_dir")
    parser = build_stage_flag_parser((spec,))

    assert isinstance(parser, argparse.ArgumentParser)

    args = parser.parse_args(["--exp-dir", "/tmp/exp"])
    assert args.exp_dir == "/tmp/exp"


# ---------------------------------------------------------------------------
# Collision detection: errors surface at parser-build time
# ---------------------------------------------------------------------------


def test_add_stage_flags_surface_flag_collision():
    """Flag name collision raised when building parser."""
    spec1 = StageSpec(name="train", config="training", log_dir="stats_dir")
    spec2 = StageSpec(
        name="decode",
        config="inference",
        log_dir="training_dir",
        dir_aliases=("stats_dir",),  # Makes --stats-dir collide
    )
    parser = argparse.ArgumentParser()

    with pytest.raises(StageContractError, match="CLI flag name collision"):
        _ = add_stage_flags(parser, (spec1, spec2))


# ---------------------------------------------------------------------------
# Flag format: dashed option strings
# ---------------------------------------------------------------------------


def test_add_stage_flags_creates_dashed_flags():
    """Flags use dashes in CLI, underscores in dest."""
    spec = StageSpec(name="train", config="training", log_dir="output_path")
    parser = argparse.ArgumentParser()
    parser = add_stage_flags(parser, (spec,))

    option_strings = [
        action.option_strings[0] for action in parser._actions if action.option_strings
    ]
    assert "--output-path" in option_strings
