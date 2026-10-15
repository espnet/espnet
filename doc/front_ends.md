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

ESPnet3 recipes construct these native inference classes through their
provider. The Runner passes the configured dataset columns directly to the
backend and uses the recipe's output formatter to produce SCP records.

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
- a tool function named after the verb, whose docstring describes the tool: an
  `Args:` line per argument and a `Returns:` that says what comes back and
  roughly what it costs. An agent reads that before it calls anything, so a
  caveat belongs there rather than in the returned text
- audio arrives as a path on this machine and nothing is uploaded
- anything a caller can fix is a `ToolError` with the fix in the message
- registration in `build_server`, and a case in `test_mcp_server.py`

Keep the tool signature to what the task needs. `phonemize` takes audio and a
language, not a prompt, because a second input would make the simplest thing
harder to call — the browser is where a second input costs nothing.

### Notebooks and hosted applications

Notebooks and applications call the native inference backend directly.
Their input conversion and output presentation belong to that application.
ESPnet3 does not generate a notebook or an inference UI from declarations.
