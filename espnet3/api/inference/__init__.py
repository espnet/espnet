"""The inference contract: what a trained ESPnet3 model promises a caller.

Every system under ``espnet3/systems/<name>/`` ships an ``inference.py``
whose ``Inference`` class subclasses :class:`InferenceAPI`. The class
declares the fields it takes and returns and implements one hook -
:meth:`InferenceAPI.run_stream` if the model works online,
:meth:`InferenceAPI.run` if it needs its whole input - and the base class
does everything a caller should not have to think about: reading a file,
accepting what Gradio hands over, resampling to the model's rate, checking
the declared fields, deriving the other hook. So the command line, the MCP
server, a Space, a notebook and the ``infer`` stage can all drive any
system the same way.

Three ways to call a loaded model::

    >>> from espnet3.api.inference import load
    >>> model = load("espnet/some_pack")           # meta.yaml names the system
    >>> model("utt.wav")["text"]                   # one shot, from a file
    >>> model(speech=(16000, samples))["text"]     # one shot, from gr.Audio
    >>> for piece in model.stream(microphone_chunks()):
    ...     print(piece.get("text", ""), end="")   # online
    >>> model.batch([{"speech": a}, {"speech": b}])  # several at once

Inference is a stream of chunks in and chunks out; the one-shot call is
the stream of one chunk, and a batch is several one-shot calls that a
model may choose to run together.

What a field can hold is a :class:`Kind` registered in :data:`KINDS`
(``espnet3.api.inference.kinds``); ``audio``, ``text``, ``segments`` and
``number`` are built in, and a new modality is one subclass passed to
:func:`register_kind` - from a system, or from a recipe's own ``src/``.

This package is for whoever calls a model. It asks nothing of the rest
of ESPnet3 - no recipe, no stage, no dataset, no cluster - and depends on
none of it; a user who has one file and a model tag needs to read nothing
else. Its names say what each thing is for - :class:`InferenceAPI`,
:class:`Kind`, :class:`Field`. (For ESPnet3 developers: the ``Base*``
spelling of an abstract class belongs to the training and ``infer``
stage machinery and stops at this package, which that machinery adapts
to, never the reverse.)

The package is laid out by what a reader looks for: :mod:`.base` holds
:class:`InferenceAPI`, :mod:`.field` the :class:`Field` declaration,
:mod:`.kinds` the kinds, and :mod:`.loading` :func:`load`. Everything is
importable from here.

A system says nothing about *what task* it performs; it says what goes in
and what comes out. A front end that offers ``transcribe`` looks for a
model whose first input is audio and whose outputs hold ``text``, and a
model that answers a conversation declares one ``messages`` field and no
list of the tasks a prompt might ask of it.

Relation to the provider and runner
-----------------------------------

ESPnet3 runs its stages through an ``EnvironmentProvider`` and a
``BaseRunner`` (``espnet3/parallel``): the provider builds the objects a
stage needs, the runner processes dataset shards, in parallel, with
resume and writers. The contract and that pair divide the work like this:

1. ``Inference`` is the only thing a system must provide for inference.
   It knows nothing of datasets, shards or runners, so the front ends -
   which have none - can use it, and so can a test.
2. The ``infer`` stage runs an ``Inference`` through the pair, and needs
   nothing added to it: ``inference.yaml``'s ``model._target_`` names the
   class (its constructor takes the model's own arguments, plus
   ``device``); ``InferenceProvider.build_model`` builds it through this
   package's :func:`build_model` - the provider calls the api, and the api
   never calls a provider back; ``InferenceRunner`` calls
   ``model(**fields)`` for one item or ``model.batch(items)`` for several,
   and writes each declared output by its kind. There is no ``output_fn``:
   the declaration fixes the outputs, and the reference for scoring is
   read from the data by ``measure`` (``ref_key: dataset:text``).
3. Parallelism - shards, workers, resume, writers - belongs to the
   runner. Decoding several items together belongs to :meth:`run_batch`.
   Streaming belongs to :meth:`run_stream`; the runner never streams.
4. System authors do not subclass the provider or runner to implement
   inference. They subclass them only for what the pair is for: how a
   dataset is built, or how outputs are written.
5. A system may not use the pair at all - a SpeechLM served by vLLM, a
   model behind an endpoint - and still provides ``Inference``, whose
   ``from_pretrained`` takes whatever handle it needs (a URL, a name),
   given to :func:`load` with ``system=`` since there is no bundle. It
   need not have an ``infer`` stage config; if it has one, it may run it
   its own way. The front ends only ever see ``Inference``.
6. Nothing assumes a task. No layer - the contract, a kind, the
   provider/runner, a front end - may ask a system what task it performs
   or dispatch on a task name; what a model does is read from its fields.
   A verb such as ``transcribe`` is a front end's word for a shape of
   fields, and a multi-task model declares its fields once.
"""

from __future__ import annotations

from espnet3.api.inference.base import InferenceAPI, check_contract, gather
from espnet3.api.inference.field import Field
from espnet3.api.inference.kinds import (
    KINDS,
    Audio,
    AudioKind,
    Kind,
    NumberKind,
    SegmentsKind,
    TextKind,
    register_kind,
)
from espnet3.api.inference.loading import (
    SYSTEM_ALIASES,
    ModelTagError,
    apply_overrides,
    build_model,
    load,
    load_model,
    locate_pack,
    read_bundle,
    read_meta,
)

__all__ = [
    "KINDS",
    "SYSTEM_ALIASES",
    "Audio",
    "AudioKind",
    "InferenceAPI",
    "Kind",
    "Field",
    "ModelTagError",
    "NumberKind",
    "SegmentsKind",
    "TextKind",
    "apply_overrides",
    "build_model",
    "check_contract",
    "gather",
    "load",
    "load_model",
    "locate_pack",
    "read_bundle",
    "read_meta",
    "register_kind",
]
