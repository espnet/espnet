# `espnet3/cli/`

See [`.agent/CLAUDE.md`](../../CLAUDE.md) for cross-cutting guidance.

```
espnet3/cli/
├── main.py                 # argparse dispatcher, registers subcommands
└── clone/                  # `espnet3 clone` subcommand
    ├── command.py          # add_arguments() / run() -- copies + rewrites publication/demo config
    └── resolver.py         # resolve_recipe() / list_recipes() -- <dataset>/<task> -> egs3/ path
```

The `espnet3` console-script entry point (`espnet3/cli/main.py:main`). Dispatches to subcommands
registered under `espnet3/cli/<name>/`; today the only subcommand is `clone`.

- **`clone/resolver.py`** -- `resolve_recipe("<dataset>/<task>")` maps a recipe identifier to its path
  inside `egs3/` (raises `ValueError` for a malformed identifier, `FileNotFoundError` -- with the full
  recipe list -- if it does not exist). `list_recipes()` enumerates every `<dataset>/<task>` directory
  under `egs3/`, skipping `TEMPLATE` and anything starting with `.`/`_`.
- **`clone/command.py`** -- `espnet3 clone <dataset>/<task> [--project DIR] [--list]`. Copies
  `conf/`, `src/`, `dataset/`, `run.py`, `readme.md`, `path.sh` into a fresh directory (refuses to
  overwrite an existing one), preserving in-recipe symlinks with relative targets and dereferencing
  symlinks that point outside the recipe. Afterwards it injects a per-clone Hugging Face repo name
  into the cloned `publication.yaml` / `demo.yaml` (`_inject_corpus_system`). The clone is a plain
  directory, not a Python package (no `__init__.py`), and only needs `espnet3` installed to run.

See [`.agent/egs3/CLAUDE.md`](../../egs3/CLAUDE.md) for the "clone and keep working in it" workflow
this subcommand exists to support, and for what a cloned recipe's `dataset/`/`src/` are expected to
contain.
