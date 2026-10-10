# Installation troubleshooting

::: important
If you hit dependency problems, prefer a clean Pixi environment first.
Most Python and non-Python dependency conflicts are easier to isolate there.
:::

## Basic checks

```bash
python -c "import espnet3"
python -c "from espnet3.systems.asr.system import ASRSystem"
```

## Common failures

- CUDA or PyTorch mismatch: reinstall PyTorch first, then reinstall `espnet`.
- Missing optional package: install the matching extra, e.g. `uv pip install -e ".[asr]"`
  (also `"[tts]"`, `"[enh]"`, ...).
- Hugging Face upload missing tools: install `huggingface_hub` and log in via
  Python (`from huggingface_hub import login; login()`) or via the CLI with
  `hf login`.

::: warning
Do not debug a broken environment by layering random `pip install` commands on
top of an old setup.
Start from a clean environment when possible.
:::

## Other dependency-related failures

### ffmpeg, cmake, etc

If you hit a version conflict or installation issue with non-Python
dependencies such as `ffmpeg` or `cmake`, we strongly recommend using
[pixi](https://pixi.prefix.dev/latest/).

```shell
curl -fsSL https://pixi.sh/install.sh | bash
pixi init && pixi add python=3.11 pip ffmpeg
pixi shell  # or: eval "$(pixi shell-hook)"
uv pip install -e .
uv pip install -e ".[asr]"    # add extras as needed, e.g. also "[tts]" or "[enh]"
```

## When to report an issue

Open an issue after you can show:

- your install command
- Python version
- PyTorch version
- the full traceback

::: note
Issue reports are much easier to handle when they include the exact install
commands and the first failing traceback.
:::

::: tip Help us expand this page
If you hit an installation error not listed here, open an issue — we will add
it to this page so the next person doesn't have to debug it from scratch.
:::

## Need more help?

<DocCards :cols="2">
  <DocCard
    title="Search existing issues"
    desc="Check whether the same install error has already been reported."
    icon="tabler:search"
    href="https://github.com/espnet/espnet/issues?q=is%3Aissue"
    :external="true"
  />
  <DocCard
    title="Open a new issue"
    desc="Report a new problem after you collect the install details above."
    icon="tabler:message-report"
    href="https://github.com/espnet/espnet/issues/new/choose"
    :external="true"
  />
</DocCards>
