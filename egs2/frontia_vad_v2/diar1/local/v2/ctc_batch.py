"""CTC segmentation of many recordings at once with an OWSM-CTC aligner.

espnet2.bin.s2t_align.CTCSegmentation aligns one recording at a time: it cuts
the recording into fixed-length buffers (the model's training length, with a
context on each side), runs the encoder on them, and keeps the middle frames
of each buffer. Since every buffer has the same length, the buffers of many
recordings can share one encoder forward without padding. This module does
that, keeps for each recording only the posterior columns of its transcript's
tokens and the blank (the rest of the 50k vocabulary is never read by CTC
segmentation), and runs the segmentation of the recordings in a process pool
while the GPU encodes the next batch. The results are those of
CTCSegmentation.__call__ for each recording.
"""

import collections
import copy
import logging
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch

from espnet2.bin.s2t_align import buffered_frame_counts, train_preprocessor_conf
from espnet2.tasks.s2t_ctc import S2TTask
from espnet2.torch_utils.device_funcs import to_device


class BufferGeometry:
    """How get_lpz() of the aligner cuts a recording into encoder buffers."""

    def __init__(self, aligner):
        conf = train_preprocessor_conf(S2TTask, aligner.s2t_train_args)
        buffer_secs = conf["speech_length"]
        context_secs = aligner.context_len_in_secs
        fps = aligner.frames_per_sec
        if context_secs is None:
            context_secs = round(4 * fps) / fps
        buffer_frames, self.context_frames = buffered_frame_counts(
            fps, buffer_secs, context_secs
        )
        self.chunk_frames = buffer_frames - 2 * self.context_frames
        fs = aligner.fs
        self.buffer_len = round(fs * buffer_secs)
        self.chunk_len = round(fs * (buffer_secs - 2 * context_secs))
        self.pad = round(fs * context_secs)
        self.buffer_secs = buffer_secs

    def buffers(self, speech: np.ndarray) -> Tuple[np.ndarray, int]:
        """(buffers x samples, samples the posteriors cover), as get_lpz() cuts."""
        speech = np.pad(speech, (self.pad, self.pad))
        out = []
        for i in range(0, len(speech), self.chunk_len):
            cur = speech[i : i + self.buffer_len]
            if len(cur) < self.buffer_len:
                out.append(np.pad(cur, (0, self.buffer_len - len(cur))))
                break
            out.append(cur)
        return np.array(out), len(out) * self.chunk_len


@torch.no_grad()
def encode(
    aligner,
    geometry: BufferGeometry,
    buffers: torch.Tensor,
    prefix: torch.Tensor,
    spent: Dict[str, float] = None,
):
    """CTC log posteriors (buffers, chunk_frames, vocabulary) of a batch of buffers.

    The same steps as get_lpz() of the aligner, for buffers of any recordings;
    `prefix` holds the (language, task) token ids of each buffer. `spent`
    collects the seconds of the host-to-device copy and of the forward.
    """
    spent = {} if spent is None else spent
    model = aligner.s2t_model
    n = buffers.size(0)
    text_prev = torch.full((n, 1), model.na, dtype=torch.long)
    batch = {
        "speech": buffers.to(getattr(torch, aligner.dtype)),
        "speech_lengths": torch.full((n,), buffers.size(1), dtype=torch.long),
        "text_prev": text_prev,
        "text_prev_lengths": torch.ones(n, dtype=torch.long),
        "prefix": prefix,
        "prefix_lengths": torch.full((n,), prefix.size(1), dtype=torch.long),
    }
    t0 = time.perf_counter()
    batch = to_device(batch, device=aligner.device)
    _sync(batch["speech"])
    t1 = time.perf_counter()
    enc, _ = model.encode(**batch)
    if isinstance(enc, tuple):
        enc = enc[0]
    enc = model.frames(enc, batch["prefix"])
    c, k = geometry.context_frames, geometry.chunk_frames
    if enc.size(1) < c + k:
        raise ValueError("context is too short for the encoder's edge loss")
    out = aligner.ctc.log_softmax(enc[:, c : c + k])
    _sync(out)
    spent["h2d"] = spent.get("h2d", 0.0) + t1 - t0
    spent["forward"] = spent.get("forward", 0.0) + time.perf_counter() - t1
    return out


def encode_or_split(aligner, geometry, buffers, prefix, spent=None):
    """encode(), and if the GPU runs out of memory, the two halves separately.

    Each buffer is encoded independently of the others in its batch, so the
    halves give the posteriors of the whole batch.
    """
    try:
        return encode(aligner, geometry, buffers, prefix, spent)
    except torch.cuda.OutOfMemoryError:
        if buffers.size(0) == 1:
            raise
        torch.cuda.empty_cache()
        half = buffers.size(0) // 2
        logging.warning("out of GPU memory at %d buffers: halves", buffers.size(0))
        return torch.cat(
            [
                encode_or_split(
                    aligner, geometry, buffers[:half], prefix[:half], spent
                ),
                encode_or_split(
                    aligner, geometry, buffers[half:], prefix[half:], spent
                ),
            ]
        )


def _sync(x: torch.Tensor):
    if x.is_cuda:
        torch.cuda.synchronize(x.device)


def reduce_task(task, lpz: torch.Tensor, spent: Dict[str, float] = None):
    """The task with posteriors of only its tokens and the blank.

    CTC segmentation reads the posteriors of the ground-truth tokens and of
    the blank only, so the other columns are dropped and the token ids of the
    ground truth (and the blank and character list of the config) renumbered.
    """
    gt = task.ground_truth_mat
    # a shallow copy: only blank and char_list change, and they are replaced
    config = copy.copy(task.config)
    ids = sorted({int(i) for i in np.unique(gt) if i >= 0} | {config.blank})
    new = np.full(max(ids) + 1, -1, dtype=np.int64)
    new[ids] = np.arange(len(ids))
    task.ground_truth_mat = np.where(gt >= 0, new[np.maximum(gt, 0)], -1)
    config.blank = int(new[config.blank])
    config.char_list = [config.char_list[i] for i in ids]
    task.config = config
    t0 = time.perf_counter()
    cols = torch.tensor(ids, device=lpz.device)
    task.lpz = lpz.index_select(-1, cols).float().cpu().numpy()
    if spent is not None:
        spent["gather_d2h"] = spent.get("gather_d2h", 0.0) + time.perf_counter() - t0
    return task


def token_word_spans(timings, utt_begin_indices, index_duration: float):
    """(start, end) in seconds of each word, from the frames of its own tokens.

    ``timings[c]`` is the time of the frame at which the CTC segmentation path
    enters ground-truth position ``c``; the tokens of word ``i`` are the
    positions between its separators ``utt_begin_indices[i]`` and
    ``utt_begin_indices[i + 1]``. A word spans from the frame of its first
    token to the end of the frame of its last token.

    ctc_segmentation.determine_utterance_segments places the boundaries of a
    segment between the separators instead: a start up to 0.5 s before the
    first token and an end at the last token, which shifts single words early
    and closes the pauses between them.
    """
    spans = []
    for i in range(len(utt_begin_indices) - 1):
        first, last = utt_begin_indices[i] + 1, utt_begin_indices[i + 1] - 1
        if last < first:
            raise ValueError(f"word {i} has no token")
        spans.append((float(timings[first]), float(timings[last]) + index_duration))
    return spans


class TooFewFrames(ValueError):
    """The emission has fewer frames than the targets need (CTC forced alignment)."""


def word_token_ids(task) -> List[List[int]]:
    """The token ids of each word of a segmentation task, in the model's vocabulary.

    The ground truth holds the tokens of word ``i`` between its separators
    ``utt_begin_indices[i]`` and ``utt_begin_indices[i + 1]`` (column 0).
    """
    gt, ubi, blank = task.ground_truth_mat, task.utt_begin_indices, task.config.blank
    return [
        [
            int(gt[c, 0])
            for c in range(ubi[i] + 1, ubi[i + 1])
            if gt[c, 0] not in (-1, blank)
        ]
        for i in range(len(ubi) - 1)
    ]


def forced_align_word_spans(lpz, words: List[List[int]], blank: int, index_duration):
    """(start, end, score) of each word by CTC forced alignment (torchaudio).

    lpz: (frames, vocabulary) log posteriors; words: the token ids of each
    word. torchaudio.functional.forced_align gives the best path of the whole
    token sequence, merge_tokens its token spans; a word spans from the first
    frame of its first token to the end of the last frame of its last token,
    in seconds (frame times are index_duration apart, as in CTC segmentation).
    Its score is the mean log posterior of its token frames. Raises
    TooFewFrames when the frames cannot hold the tokens (forced_align needs
    frames >= tokens + consecutive repeats).
    """
    import torch
    import torchaudio.functional as taf

    flat = [t for w in words for t in w]
    if any(not w for w in words):
        raise ValueError("a word has no token")
    repeats = sum(a == b for a, b in zip(flat, flat[1:]))
    if lpz.size(0) < len(flat) + repeats:
        raise TooFewFrames(
            f"{lpz.size(0)} frames < {len(flat)} tokens + {repeats} repeats"
        )
    targets = torch.tensor([flat], dtype=torch.int32, device=lpz.device)
    paths, scores = taf.forced_align(lpz[None].float(), targets, blank=blank)
    token_spans = taf.merge_tokens(paths[0], scores[0], blank=blank)
    assert len(token_spans) == len(flat), (len(token_spans), len(flat))
    out, k = [], 0
    for w in words:
        spans = token_spans[k : k + len(w)]
        k += len(w)
        frames = sum(sp.end - sp.start for sp in spans)
        score = sum(sp.score * (sp.end - sp.start) for sp in spans) / frames
        out.append(
            (spans[0].start * index_duration, spans[-1].end * index_duration, score)
        )
    return out


def _segments(task):
    """Run in a worker: the segments and token word spans of one task.

    get_segments + task.set, then token_word_spans.
    """
    from espnet2.bin.s2t_align import CTCSegmentation

    t0 = time.perf_counter()
    try:
        task.set(**CTCSegmentation.get_segments(task))
        segments = [tuple(map(float, s)) for s in task.segments]
        tokens = token_word_spans(
            task.timings, task.utt_begin_indices, task.config.index_duration_in_seconds
        )
        return task.name, segments, tokens, None, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - reported per recording, as one by one
        error = f"{type(e).__name__}: {e}"
        return task.name, None, None, error, time.perf_counter() - t0


def align_batched(
    aligner,
    items: Iterable[Tuple[str, List[str], str, Callable[[], Tuple[int, np.ndarray]]]],
    batch_seconds: float,
    workers: int,
    prefetch: int = 64,
    forced_align: bool = False,
) -> Dict[str, Tuple[list, dict, str]]:
    """{rec_id: (segments, spans, error)} of (rec_id, words, lang_sym, read).

    spans maps "tokens" to the word spans of token_word_spans and, with
    forced_align, "forced_align" to those of forced_align_word_spans (or to
    an error string when that recording cannot be force-aligned). segments
    and spans are None when error is not None.

    `read` returns (rate, samples) of the recording; it runs in a reader
    thread, `prefetch` recordings ahead. Buffers are encoded `batch_seconds`
    of audio (in buffers of the model's length) at a time. A second thread
    turns each encoded batch into segmentation tasks (tokenization, the
    posterior columns of each recording, copy to the host) while the GPU
    encodes the next one; the segmentation runs in `workers` processes.
    """
    import multiprocessing

    geometry = BufferGeometry(aligner)
    per_batch = max(1, int(batch_seconds // geometry.buffer_secs))
    task_sym = aligner.token_list.index(aligner.task_sym)
    results: Dict[str, Tuple[list, dict, str]] = {}
    fa_spans: Dict[str, object] = {}  # forced alignment of each recording
    inq: "queue.Queue" = queue.Queue(maxsize=prefetch)
    postq: "queue.Queue" = queue.Queue(maxsize=2)  # encoded batches
    futures = []
    # seconds per phase, logged per batch and at the end
    spent: Dict[str, float] = collections.defaultdict(float)
    wall0 = time.perf_counter()

    def reader():
        for rid, words, lang_sym, read in items:
            try:
                inq.put((rid, words, lang_sym, read(), None))
            except Exception as e:  # noqa: BLE001 - an unreadable recording
                inq.put((rid, words, lang_sym, None, f"{type(e).__name__}: {e}"))
        inq.put(None)

    def post(pool):
        while True:
            item = postq.get()
            if item is None:
                return
            pending, lpz_all = item
            start = 0
            for rid, words, _, bufs, covered in pending:
                lpz = lpz_all[start : start + len(bufs)].reshape(-1, lpz_all.size(-1))
                start += len(bufs)
                t0 = time.perf_counter()
                try:
                    placeholder = np.zeros((lpz.size(0), 1), dtype=np.float32)
                    task = aligner.prepare_segmentation_task(
                        words, placeholder, name=rid, speech_len=covered
                    )
                    spent["prepare"] += time.perf_counter() - t0
                    if forced_align:
                        t1 = time.perf_counter()
                        try:
                            fa_spans[rid] = forced_align_word_spans(
                                lpz,
                                word_token_ids(task),
                                task.config.blank,
                                task.config.index_duration_in_seconds,
                            )
                        except TooFewFrames as e:
                            fa_spans[rid] = f"TooFewFrames: {e}"
                        spent["forced_align"] += time.perf_counter() - t1
                    task = reduce_task(task, lpz, spent)
                    futures.append(pool.submit(_segments, task))
                except Exception as e:  # noqa: BLE001 - e.g. no token of the transcript
                    results[rid] = (None, None, f"{type(e).__name__}: {e}")

    def flush(pending):
        if not pending:
            return
        rows = np.concatenate([p[3] for p in pending])
        prefix = torch.tensor(
            [[p[2], task_sym] for p in pending for _ in range(len(p[3]))],
            dtype=torch.long,
        )
        outs = [
            encode_or_split(
                aligner,
                geometry,
                torch.from_numpy(rows[i : i + per_batch]),
                prefix[i : i + per_batch],
                spent,
            )
            for i in range(0, len(rows), per_batch)
        ]
        t0 = time.perf_counter()
        postq.put((list(pending), torch.cat(outs)))
        spent["post_wait"] += time.perf_counter() - t0
        logging.info(
            "batch of %d buffers; so far (s): %s",
            len(rows),
            ", ".join(f"{k} {v:.0f}" for k, v in sorted(spent.items())),
        )

    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        threading.Thread(target=reader, daemon=True).start()
        poster = threading.Thread(target=post, args=(pool,))
        poster.start()
        pending, queued = [], 0
        while True:
            t0 = time.perf_counter()
            item = inq.get()
            spent["read_wait"] += time.perf_counter() - t0
            if item is None:
                break
            rid, words, lang_sym, audio, error = item
            if error is not None:
                results[rid] = (None, None, error)
                continue
            rate, x = audio
            if rate != aligner.fs:
                results[rid] = (None, None, f"rate {rate} is not {aligner.fs}")
                continue
            bufs, covered = geometry.buffers(np.asarray(x, dtype=np.float32))
            lang_id = aligner.token_list.index(lang_sym)
            pending.append((rid, words, lang_id, bufs, covered))
            queued += len(bufs)
            if queued >= per_batch:
                flush(pending)
                pending, queued = [], 0
        flush(pending)
        postq.put(None)
        poster.join()
        t0 = time.perf_counter()
        for f in futures:
            rid, segments, tokens, error, dp = f.result()
            spans = None
            if error is None:
                spans = {"tokens": tokens}
                if forced_align:
                    spans["forced_align"] = fa_spans.get(rid, "not aligned")
            results[rid] = (segments, spans, error)
            spent["dp_in_workers"] += dp
        spent["dp_wait"] += time.perf_counter() - t0
    wall = time.perf_counter() - wall0
    logging.info(
        "time: wall %.0f s; %s; GPU busy %.0f %% of the wall",
        wall,
        ", ".join(
            f"{k} {v:.0f} s ({v / wall * 100:.0f} %)" for k, v in sorted(spent.items())
        ),
        spent["forward"] / wall * 100,
    )
    return results
