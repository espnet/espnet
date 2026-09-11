# Dev Setup

::: important
ESPnet3 development is moving to a Pixi-based workflow.
New contributor setup should start from Pixi instead of conda.
:::

## Why Pixi

ESPnet3 development is moving to a Pixi-based workflow.

Pixi is a good base for ESPnet3 because it can:

- replace conda for Python and non-Python dependencies
- install tools such as `ffmpeg` and `cmake` in the same environment
- work well with `uv` for fast Python package installs

If you install Pixi, you usually do not need a separate conda setup.

::: note
Pixi handles Python and non-Python dependencies together, while `uv` gives fast
Python package installs inside that environment.
:::

## Install Pixi

Install Pixi first:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Open a new shell after the install, or reload your shell config.

## Create a development environment

Clone ESPnet and move into the repository:

```bash
git clone https://github.com/espnet/espnet.git
cd espnet
```

Create a Pixi environment with Python and common system tools:

```bash
pixi init
pixi add python=3.11 pip ffmpeg
```

If your recipe needs extra system packages, add them in the same way:

```bash
pixi add sox cmake
```

Then enter the environment:

```bash
pixi shell
```

If you want to enter the shell from shell script, you can write:

```bash
eval "$(pixi shell-hook)"
```

## Install ESPnet3 for development

Inside the Pixi shell, install ESPnet from source:

```bash
uv pip install -e .
```

Add more extras if needed:

```bash
uv pip install -e ".[asr,tts]"
uv pip install -e ".[enh]"
```

This keeps Python package installation fast while Pixi manages the base
environment.

::: tip If something goes wrong
If the installation does not behave as expected, explicitly passing the Python
interpreter often fixes it:

```bash
uv pip install -e . --python $(which python)
```
:::

## Install extra Python packages

For extra Python packages, use `uv` inside the Pixi shell. For example:

```bash
uv pip install lightning transformers gradio
```

If you need a package from a local checkout:

```bash
uv pip install -e /path/to/package
```

## Install non-Python dependencies

Use `pixi add` for tools that are not Python packages.
Package names follow conda-forge naming, not apt or brew:

```bash
pixi add ffmpeg cmake sox
```

You can check that they are available from the Pixi environment:

```bash
pixi shell
ffmpeg -version
cmake --version
```

This is the preferred way to install common recipe dependencies.


## Optional recipe tool installers

Some recipes still need extra installers from `tools/installers/`.
Use them after entering the Pixi shell:

```bash
cd tools
./installers/<installer>.sh
```

Examples:

- `install_ffmpeg.sh`
- `install_k2.sh`
- `install_warp-transducer.sh`

Check the script before running it.

::: warning
Most installers call `pip` directly rather than `uv`, which can be slower and
may install packages into an unexpected environment.
If you hit issues, replace the `pip install` calls inside the script with
`uv pip install` for faster, more predictable installs.
:::

## Verify the setup

```bash
python -c "import espnet3"
python -c "from espnet3.systems.asr.system import ASRSystem"
ffmpeg -version
```

## Run tests

```bash
pytest test/espnet3/
```

Run a smaller test target when possible:

```bash
pytest test/espnet3/systems/asr/
```

## Troubleshooting

- `pixi: command not found`: restart your shell after installing Pixi.
- `uv: command not found`: add `uv` with `pixi add uv`, then reopen `pixi shell`.
- CUDA or PyTorch mismatch: reinstall the correct PyTorch build inside the Pixi environment.
- Missing recipe dependency: check `tools/installers/` and the recipe README.

If the environment still fails, include these in your issue:

- the `pixi add` command you used
- the `uv pip install` command you used
- `python --version`
- the full traceback
