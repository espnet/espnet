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
    aligner, geometry: BufferGeometry, buffers: torch.Tensor, prefix: torch.Tensor
):
    """CTC log posteriors (buffers, chunk_frames, vocabulary) of a batch of buffers.

    The same steps as get_lpz() of the aligner, for buffers of any recordings;
    `prefix` holds the (language, task) token ids of each buffer.
    """
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
    batch = to_device(batch, device=aligner.device)
    enc, _ = model.encode(**batch)
    if isinstance(enc, tuple):
        enc = enc[0]
    enc = model.frames(enc, batch["prefix"])
    c, k = geometry.context_frames, geometry.chunk_frames
    if enc.size(1) < c + k:
        raise ValueError("context is too short for the encoder's edge loss")
    return aligner.ctc.log_softmax(enc[:, c : c + k])


def reduce_task(task, lpz: torch.Tensor):
    """The task with posteriors of only its tokens and the blank.

    CTC segmentation reads the posteriors of the ground-truth tokens and of
    the blank only, so the other columns are dropped and the token ids of the
    ground truth (and the blank and character list of the config) renumbered.
    """
    gt = task.ground_truth_mat
    config = copy.deepcopy(task.config)
    ids = sorted({int(i) for i in np.unique(gt) if i >= 0} | {config.blank})
    new = np.full(max(ids) + 1, -1, dtype=np.int64)
    new[ids] = np.arange(len(ids))
    task.ground_truth_mat = np.where(gt >= 0, new[np.maximum(gt, 0)], -1)
    config.blank = int(new[config.blank])
    config.char_list = [config.char_list[i] for i in ids]
    task.config = config
    cols = torch.tensor(ids, device=lpz.device)
    task.lpz = lpz.index_select(-1, cols).float().cpu().numpy()
    return task


def _segments(task):
    """Run in a worker: the segments of one task (get_segments + task.set)."""
    from espnet2.bin.s2t_align import CTCSegmentation

    try:
        task.set(**CTCSegmentation.get_segments(task))
        return task.name, [tuple(map(float, s)) for s in task.segments], None
    except Exception as e:  # noqa: BLE001 - reported per recording, as one by one
        return task.name, None, f"{type(e).__name__}: {e}"


def align_batched(
    aligner,
    items: Iterable[Tuple[str, List[str], str, Callable[[], Tuple[int, np.ndarray]]]],
    batch_seconds: float,
    workers: int,
    prefetch: int = 64,
) -> Dict[str, Tuple[list, str]]:
    """{rec_id: (segments or None, error or None)} of (rec_id, words, lang_sym, read).

    `read` returns (rate, samples) of the recording; it runs in a reader
    thread, `prefetch` recordings ahead. Buffers are encoded `batch_seconds`
    of audio (in buffers of the model's length) at a time; the segmentation
    runs in `workers` processes.
    """
    geometry = BufferGeometry(aligner)
    per_batch = max(1, int(batch_seconds // geometry.buffer_secs))
    task_sym = aligner.token_list.index(aligner.task_sym)
    results: Dict[str, Tuple[list, str]] = {}
    q: "queue.Queue" = queue.Queue(maxsize=prefetch)

    def reader():
        for rid, words, lang_sym, read in items:
            try:
                q.put((rid, words, lang_sym, read(), None))
            except Exception as e:  # noqa: BLE001 - an unreadable recording
                q.put((rid, words, lang_sym, None, f"{type(e).__name__}: {e}"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    futures = []
    pending: list = []  # (rid, words, lang_id, buffers, covered)
    # seconds spent waiting for audio, encoding, preparing tasks, and waiting
    # for the segmentation at the end (logged; tells which side is the limit)
    spent = {"read_wait": 0.0, "encode": 0.0, "prepare": 0.0, "segment_wait": 0.0}

    def flush(pool):
        if not pending:
            return
        rows = np.concatenate([p[3] for p in pending])
        prefix = torch.tensor(
            [[p[2], task_sym] for p in pending for _ in range(len(p[3]))],
            dtype=torch.long,
        )
        t0 = time.perf_counter()
        outs = [
            encode(
                aligner,
                geometry,
                torch.from_numpy(rows[i : i + per_batch]),
                prefix[i : i + per_batch],
            )
            for i in range(0, len(rows), per_batch)
        ]
        lpz_all = torch.cat(outs)
        if lpz_all.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        spent["encode"] += t1 - t0
        start = 0
        for rid, words, _, bufs, covered in pending:
            lpz = lpz_all[start : start + len(bufs)].reshape(-1, lpz_all.size(-1))
            start += len(bufs)
            try:
                placeholder = np.zeros((lpz.size(0), 1), dtype=np.float32)
                task = aligner.prepare_segmentation_task(
                    words, placeholder, name=rid, speech_len=covered
                )
                futures.append(pool.submit(_segments, reduce_task(task, lpz)))
            except Exception as e:  # noqa: BLE001 - e.g. no token of the transcript
                results[rid] = (None, f"{type(e).__name__}: {e}")
        spent["prepare"] += time.perf_counter() - t1
        pending.clear()

    import multiprocessing

    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        queued = 0
        while True:
            t0 = time.perf_counter()
            item = q.get()
            spent["read_wait"] += time.perf_counter() - t0
            if item is None:
                break
            rid, words, lang_sym, audio, error = item
            if error is not None:
                results[rid] = (None, error)
                continue
            rate, x = audio
            if rate != aligner.fs:
                results[rid] = (None, f"rate {rate} is not {aligner.fs}")
                continue
            bufs, covered = geometry.buffers(np.asarray(x, dtype=np.float32))
            lang_id = aligner.token_list.index(lang_sym)
            pending.append((rid, words, lang_id, bufs, covered))
            queued += len(bufs)
            if queued >= per_batch:
                flush(pool)
                queued = 0
        flush(pool)
        t0 = time.perf_counter()
        for n, f in enumerate(futures):
            rid, segments, error = f.result()
            results[rid] = (segments, error)
            if n % 1000 == 0:
                logging.info("%d / %d segmented", n, len(futures))
        spent["segment_wait"] += time.perf_counter() - t0
    logging.info("time (s): %s", ", ".join(f"{k} {v:.0f}" for k, v in spent.items()))
    return results
