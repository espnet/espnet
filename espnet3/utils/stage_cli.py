"""CLI utilities for wiring stage log_dir flags from StageSpec.

Builds argparse parsers whose arguments map to config keys via
`generate_stage_flag_mapping` from espnet3.contract.stages.
"""

from __future__ import annotations

import argparse
import re
from typing import TYPE_CHECKING

from espnet3.contract.stages import StageContractError, generate_stage_flag_mapping

if TYPE_CHECKING:
    from espnet3.contract.stages import StageSpec


# Normalize a config key or OmegaConf dotted path into a safe argparse destination.
# e.g., "tokenizer.save_path" -> "tokenizer_save_path"
#        "exp_dir" -> "exp_dir"
def _sanitize_dest_key(key: str) -> str:
    return key.replace(".", "_")


def add_stage_flags(
    parser: argparse.ArgumentParser,
    system_stages: tuple[StageSpec, ...],
    *,
    prefix: str = "",
) -> argparse.ArgumentParser:
    """Add CLI flags for all stage log_dir targets to an argparse parser.

    Transforms the output of `generate_stage_flag_mapping` into argparse
    arguments. When multiple flag names (e.g., canonical + aliases) map to
    the same canonical location, they all write their values to the SINGLE
    argparse destination corresponding to the canonical key, ensuring a
    single canonical value is captured on parse.

    Args:
        parser: The argparse parser to extend. If None, a new one is created.
        system_stages: The system's declared stages tuple.
        prefix: Prefix to prepend to all CLI flag names (e.g., "stats" for
            --stats-exp-dir-style flags). Useful when a single parser holds
            more than one "direction" of stages within the same command.
            The prefix is NOT applied to the argparse destination.

    Returns:
        The parser with new flags added.

    Examples:
        >>> import argparse
        >>> from espnet3.contract.stages import StageSpec
        >>> spec = StageSpec(name="train", config="training", log_dir="exp_dir")
        >>> parser = argparse.ArgumentParser()
        >>> parser = add_stage_flags(parser, (spec,))
        >>> args = parser.parse_args(["--exp-dir", "/tmp/exp"])
        >>> args.exp_dir
        '/tmp/exp'
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    flag_mapping = generate_stage_flag_mapping(system_stages)

    if not flag_mapping:
        # No stage log directories declared; nothing to add.
        return parser

    for flag_name, canonical_config_key in flag_mapping.items():
        # Build CLI flag name with prefix
        effective_flag = flag_name
        if prefix:
            effective_flag = f"{prefix}_{flag_name}"

        flag_name_dashed = effective_flag.replace("_", "-")

        # Use the canonical config key as the argparse destination (sanitized to identifier)
        dest_key = _sanitize_dest_key(canonical_config_key)

        help_text = f"Output directory for stage. Config key: `{canonical_config_key}`."

        parser.add_argument(
            f"--{flag_name_dashed}",
            dest=dest_key,
            type=str,
            default=None,
            help=help_text,
            metavar="DIR",
        )

    return parser


def build_stage_flag_parser(
    system_stages: tuple[StageSpec, ...],
    *,
    prefix: str = "",
) -> argparse.ArgumentParser:
    """Convenience: build a new parser with all stage log_dir flags.

    Args:
        system_stages: The system's declared stages.
        prefix: Optional prefix for flags as in :func:`add_stage_flags`.

    Returns:
        A new ArgumentParser ready to handle stage log_dir CLI arguments.

    Example:
        >>> from espnet3.contract.stages import StageSpec
        >>> spec = StageSpec(name="train", config="training", log_dir="exp_dir")
        >>> parser = build_stage_flag_parser((spec,))
        >>> parser.print_help()  # Shows --exp-dir flag
    """
    parser = argparse.ArgumentParser(
        description="Configure stage-specific directory paths."
    )
    return add_stage_flags(parser, system_stages, prefix=prefix)


__all__ = [
    "add_stage_flags",
    "build_stage_flag_parser",
]
