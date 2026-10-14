"""Compatibility CLI; new recipes use aqa_eval.py."""

from aqa_eval import (  # noqa: F401
    calculate_metrics,
    get_parser,
    load_metrics,
    load_sys_info,
    main,
)

if __name__ == "__main__":
    main()
