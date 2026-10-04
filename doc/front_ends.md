# One task, four front ends

A published model reaches a person in four ways: the command line, an MCP
server an assistant calls, a notebook, and a hosted Space. This page is how a
task gets all four, and the point of it is that the next task should cost an
afternoon rather than a design discussion.

It is the rules, and only the rules. What runs where **today** is the support
matrix in the top-level [README](https://github.com/espnet/espnet/blob/master/README.md), which `ci/check_front_ends.py`
checks against the code — so adding a task means following this page and
adding a row there, and never editing this page at all.

Everything here is a rule about **the shape**, never about what a task does.
Alignment takes two inputs and answers with a table; nothing below asks it not
to. What the rules cover is the part a reader should not have to re-learn per
task: what the thing is called, where its files live, how a model is chosen,
what a failure looks like.

## The vocabulary

**A task is a verb, and the same verb everywhere it appears**: `espnet
transcribe` at a terminal and `transcribe` as an MCP tool are one name, so a
person reading an agent's transcript and a person reading a shell history are
talking about the same thing. `asr` and `tts` were the names once and still
work, which is the only reason there are aliases at all.

Two naming systems meet here and neither wins everywhere:

- **the verb** names what a person asks for, and so names the CLI subcommand
  and the MCP tool.
- **the ESPnet task** (`asr`, `s2t`, `tts`, `enh`, `spk`, `codec`) names what
  a model was trained as, and so names recipes, `espnet.load(task=...)` and
  the notebooks — `espnet/notebook`'s `check_layout.py` enforces
  `<task>_<variant>_demo.ipynb`, which is why alignment's notebook is
  `s2t_align_demo.ipynb` and not `align_demo.ipynb`.

Say which one you are using when you name something new.

**Which tasks exist today, and how much of each is built, is the table in the
top-level [README](https://github.com/espnet/espnet/blob/master/README.md)** — not this page. This page is the rules; a
new task adds a row there and changes nothing here. `ci/check_front_ends.py`
holds that table to the code.

## Adding a task

In this order, because each step is usable before the next exists.

### 1. The inference class

Whatever `espnet2.bin` class does the work. It takes `model_tag` through
`from_pretrained`, and `test/espnet2/bin/test_from_pretrained_routing.py`
holds every such class to one way of fetching a published model.

### 2. The command line — `espnet2/bin/cli.py`

- a verb in `DEFAULT_MODELS`, pointing at the flagship checkpoint
- `cmd_<verb>`, which imports the inference stack **after** the checks a user
  can fail: a missing file should not wait for a 4 GB import
- `add("<verb>", "<one line of help>", cmd_<verb>)` in `build_parser`, which
  gives every command `--model` and `--device` for free
- a case in `test/espnet2/bin/test_cli.py`, which checks that each default tag
  loads with the class the command uses

A command is one positional input and options. If a task needs two inputs, it
is a flag (`espnet align audio.wav --text "..."`), not a second positional.

### 3. The MCP server — `espnet2/bin/mcp_server.py`

- a loader, `_x()`, wrapped in `functools.lru_cache`, so the checkpoint is
  fetched once a session
- a tool function named after the verb, whose **docstring is the API**: an
  `Args:` line per argument and a `Returns:` that says what comes back and
  roughly what it costs. An agent reads that before it calls anything, so a
  caveat belongs there rather than in the returned text
- audio arrives as a path on this machine and nothing is uploaded
- anything a caller can fix is a `ToolError` with the fix in the message
- registration in `build_server`, and a case in `test_mcp_server.py`

Keep the tool signature to what the task needs. `phonemize` takes audio and a
language, not a prompt, because a second input would make the simplest thing
harder to call — the browser is where a second input costs nothing.

### 4. The Space — `egs2/<recipe>/<task>/demo[_<variant>]/`

Three files, and a Space is exactly that directory:

- **`app.py`** — loads a model, picks a device, asks for a slice of GPU time.
  For a task whose page is "audio in, text out", **import the page** rather
  than writing one:

  ```python
  from espnet2.bin.demo import build_app
  demo = build_app(s2t, device=DEVICE, model_tag=MODEL_TAG,
                   wrap=spaces.GPU(duration=GPU_SECONDS))
  ```

  `espnet2.bin.demo` reads the menus, the window, the tasks and whether there
  is a decoder to prompt off the checkpoint, so one page serves every
  speech-to-text Space and `espnet demo`. Pass `title` and `description` for
  what the page cannot read off a model: what this checkpoint is and why
  someone would open this Space rather than the one next door. Write your own
  page only when the task's shape is different — alignment does, with two
  inputs and a table — and say why in the README.
- **`README.md`** — the Hugging Face card. `python_version: "3.12"` as a
  string, `app_file: app.py`, the model in `models:`, a
  `short_description` under 60 characters, and a section saying how to run it
  locally and when it may be uploaded.
- **`requirements.txt`** — `espnet[<extra>]>=<release>` with a lower bound,
  `espnet_model_zoo`, and what the app itself imports. The extra is the one
  that carries the dependency: `espnet[demo]` brings gradio, `espnet[tts]`,
  `espnet[enh]` and `espnet[spk]` bring the front ends those checkpoints need.

Then register the directory in `test/espnet2/bin/test_demo_apps.py::DEMOS`,
which is what checks the rest:

| rule | why |
|---|---|
| the ZeroGPU shim, word for word | it patches torch on import, so it comes before torch |
| the device rule, word for word | getting it wrong is silent: the app runs on the CPU of a machine rented for its GPU |
| the model from `<NAME>_MODEL_TAG` | a Space is switched to another checkpoint without a commit |
| the sample rate from the checkpoint | a model at another rate must not need an edit |
| one `MAX_*` cap, compared against the input and named in the message | a cap nobody is told about reads as the model losing the tail |
| `GPU_SECONDS` ≥ that cap | the slice has to cover what the page accepts |
| a pinned lowest espnet | see *Releases* below |

### 5. The notebook — `espnet/notebook`

`Demos/<task>_<variant>_demo.ipynb`, a workflow of its own for the badge, and
a row in both READMEs; `tools/check_layout.py` fails if any of those is
missing. It pins a release, opens with a `MODEL` variable so a reader can swap
checkpoints, and ends with **Where next** pointing at the other three front
ends.

### 6. The matrix

Add the row to the table in the top-level `README.md`, and say plainly what is
missing: ❌ is a cell nobody has done, 🚧 is one in review. The point of the
table is the gaps.

## Releases

A Space installs espnet from PyPI and a notebook pins a release, so **a front
end that uses something only `master` has cannot ship until the release that
carries it**. Two rules:

- a Space is uploaded *after* that release, never before — uploading early
  replaces a working Space with one that builds and then fails to start
- a notebook pins the release it needs, and its workflow sets
  `allow_unreleased_pin: true` until that release is out, so CI says which
  release it is waiting for instead of failing on the pin

`test_demo_apps.py` knows which names arrived in which release and fails a
Space whose `requirements.txt` asks for less.

## What is checked, and where

| check | file |
|---|---|
| the Spaces' shape, cards and pins | `test/espnet2/bin/test_demo_apps.py` |
| the page built from a checkpoint | `test/espnet2/bin/test_demo.py` |
| the CLI's commands and default models | `test/espnet2/bin/test_cli.py` |
| the MCP tools | `test/espnet2/bin/test_mcp_server.py` |
| `from_pretrained` on every inference class | `test/espnet2/bin/test_from_pretrained_routing.py` |
| the README's table against the CLI and the MCP tools | `ci/check_front_ends.py`, with the table |
| the links in the README | `ci/check_demo_links.py`, daily |
| the notebooks' names, badges and tables | `tools/check_layout.py` in `espnet/notebook` |
