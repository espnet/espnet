#!/usr/bin/env python3
"""Write OWSM's token list to a tokens.txt file.

OWSM ships its vocabulary inside the training config's ``token_list`` field. The
recipe reuses it unchanged (prompt-conditioning adds no new tokens), so we simply
copy it out for s2t.sh, which expects a tokens.txt when stages 5-6 are skipped.

Usage:
    python local/owsm_token_list.py <owsm_config.yaml> <out_tokens.txt>
"""

import sys

import yaml


def main() -> None:
    config_path, out_path = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(open(config_path))
    toks = cfg["token_list"]
    if isinstance(toks, str):  # a path rather than an inline list
        toks = [line.rstrip("\n") for line in open(toks)]
    with open(out_path, "w") as f:
        f.write("\n".join(toks) + "\n")
    print(f"Wrote {len(toks)} tokens to {out_path}")


if __name__ == "__main__":
    main()
