"""The inference contract: what a trained ESPnet3 model promises a caller.

Every system under ``espnet3/systems/<name>/`` ships an ``inference.py``
whose ``Inference`` class subclasses :class:`BaseInference`. The class
declares the fields it takes and returns and implements one hook -
:meth:`BaseInference.run_stream` if the model works online,
:meth:`BaseInference.run` if it needs its whole input - and the base class
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

What a field can hold is a :class:`BaseKind` registered in :data:`KINDS`
(``espnet3.api.inference.kinds``); ``audio``, ``text`` and ``segments`` are
built in, and a new modality is one subclass passed to
:func:`register_kind` - from a system, or from a recipe's own ``src/``.

The package is laid out by what a reader looks for: :mod:`.base` holds
:class:`BaseInference`, :mod:`.field` the :class:`Field` declaration,
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
   ``device``), ``InferenceProvider.build_model`` instantiates it, and
   ``InferenceRunner`` calls ``model(**fields)`` for one item or with a
   list per field for a batch, and writes the mapping it returns. The
   recipe's ``output_fn`` stays optional, for columns the contract does
   not produce, such as ``ref`` for scoring.
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

from espnet3.api.inference.base import BaseInference, check_contract, gather
from espnet3.api.inference.field import Field
from espnet3.api.inference.kinds import (
    KINDS,
    Audio,
    AudioKind,
    BaseKind,
    SegmentsKind,
    TextKind,
    register_kind,
)
from espnet3.api.inference.loading import SYSTEM_ALIASES, load, locate_pack

__all__ = [
    "KINDS",
    "SYSTEM_ALIASES",
    "Audio",
    "AudioKind",
    "BaseInference",
    "BaseKind",
    "Field",
    "SegmentsKind",
    "TextKind",
    "check_contract",
    "gather",
    "load",
    "locate_pack",
    "register_kind",
]
