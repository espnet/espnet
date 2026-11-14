#!/usr/bin/env python3
"""Speech probabilities of many models, look-aheads and test sets in one task array.

    python3 local/v2/vad_infer.py plan --out exp/eval_vad --tasks 12 \
        --model <name>=<exp dir>/<checkpoint> ... --look_aheads 0 124 \
        --sets dump/raw/<set> ...
    python3 local/v2/vad_infer.py run exp/eval_vad/plan/task.<k>.tsv \
        [--batch_seconds 14400]

plan: every (model, look-ahead, set, recording) is a work item. The items of
each (model, look-ahead) are cut into chunks of about equal audio (the
recordings longest first, round robin), and the chunks go, longest first, to
the task with the least audio; a task holds few (model, look-ahead) groups.
Writes <out>/plan/task.<k>.tsv: lines "<exp dir> <checkpoint> <look-ahead>
<set dir> <recording> <samples> <out dir>", where <out dir> is
<out>/<model>_la<look-ahead>/diar_eval/diarized_<set name>, the directory
diar.sh stage 6 would write (local/v2/score_vad.py scores it).

run: loads each (model, look-ahead) of the task once
(espnet2.bin.diar_inference.DiarizeSpeech, the model of diar.sh stage 6) and
runs its recordings sorted by length in zero-padded batches of at most
--batch_seconds of padded audio (DiarizeSpeech.diarize_batch gives every
recording what it gets alone). Writes <out dir>/predictions/<recording>.npy
(frames x 1, speech probability) and <out dir>/diarize.<task>.scp; logs the
peak GPU memory of every group.
"""

import argparse
import collections
import logging
import sys
import time
from pathlib import Path


def read_table(path: Path):
    return dict(x.split(maxsplit=1) for x in path.read_text().splitlines() if x.strip())


def cmd_plan(a):
    out = Path(a.out)
    (out / "plan").mkdir(parents=True, exist_ok=True)
    groups = collections.OrderedDict()
    for spec in a.model:
        name, path = spec.split("=", 1)
        model_file = Path(path)
        for la in a.look_aheads:
            items = []
            for s in map(Path, a.sets):
                n = {k: int(v) for k, v in read_table(s / "utt2num_samples").items()}
                for rid in read_table(s / "wav.scp"):
                    od = out / f"{name}_la{la}" / "diar_eval" / f"diarized_{s.name}"
                    items.append(
                        (
                            str(model_file.parent),
                            model_file.name,
                            la,
                            str(s),
                            rid,
                            n[rid],
                            str(od),
                        )
                    )
            groups[(name, la)] = sorted(items, key=lambda x: -x[5])
    total = sum(x[5] for g in groups.values() for x in g)
    target = total / a.tasks
    chunks = []
    for items in groups.values():
        g = sum(x[5] for x in items)
        k = max(1, round(g / target))
        parts = [[] for _ in range(k)]
        for i, x in enumerate(items):  # longest first, round robin: equal audio
            parts[i % k].append(x)
        chunks += [p for p in parts if p]
    loads = [0] * a.tasks
    tasks = [[] for _ in range(a.tasks)]
    for c in sorted(chunks, key=lambda c: -sum(x[5] for x in c)):
        k = loads.index(min(loads))
        tasks[k] += c
        loads[k] += sum(x[5] for x in c)
    for k, items in enumerate(tasks, 1):
        with (out / "plan" / f"task.{k}.tsv").open("w") as f:
            for x in items:
                f.write("\t".join(map(str, x)) + "\n")
    hours = [x / 16000 / 3600 for x in loads]
    print(
        f"{sum(len(g) for g in groups.values())} items in {len(groups)} "
        "model/look-ahead "
        f"groups, {total / 16000 / 3600:.1f} h of audio; {a.tasks} tasks: "
        f"{max(hours):.1f} h max, {min(hours):.1f} h min; groups per task max "
        f"{max(len({(x[0], x[1], x[2]) for x in t}) for t in tasks)}"
    )


def cmd_run(a):
    import numpy as np
    import torch

    from espnet2.bin.diar_inference import DiarizeSpeech
    from espnet2.fileio.omniio_audio import OmniIOAudioScpReader

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    plan = Path(a.plan)
    task = plan.name.split(".")[1]
    groups = collections.OrderedDict()
    for line in plan.read_text().splitlines():
        exp, ckpt, la, sdir, rid, n, od = line.split("\t")
        groups.setdefault((exp, ckpt, int(la)), []).append((sdir, rid, int(n), od))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    readers, scps = {}, {}
    for (exp, ckpt, la), items in groups.items():
        t0 = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = DiarizeSpeech(
            train_config=f"{exp}/config.yaml",
            model_file=f"{exp}/{ckpt}",
            num_spk=1,
            device=device,
            look_ahead=la,
        )
        items = sorted(items, key=lambda x: -x[2])
        budget = a.batch_seconds * 16000
        batches, cur = [], []
        for it in items:  # longest first: padding stays small within a batch
            if cur and (len(cur) + 1) * cur[0][2] > budget:
                batches.append(cur)
                cur = []
            cur.append(it)
        if cur:
            batches.append(cur)
        seconds = 0.0
        for batch in batches:
            wavs = []
            for sdir, rid, n, od in batch:
                if sdir not in readers:
                    readers[sdir] = OmniIOAudioScpReader(f"{sdir}/wav.scp")
                rate, x = readers[sdir][rid]
                wavs.append(np.asarray(x, dtype=np.float32))
            lengths = [len(w) for w in wavs]
            speech = np.zeros((len(wavs), max(lengths)), dtype=np.float32)
            for i, w in enumerate(wavs):
                speech[i, : len(w)] = w
            probs = model.diarize_batch(speech, np.array(lengths), fs=16000)
            for (sdir, rid, n, od), p in zip(batch, probs):
                pdir = Path(od) / "predictions"
                pdir.mkdir(parents=True, exist_ok=True)
                path = pdir / f"{rid}.npy"
                np.save(path, np.asarray(p, dtype=np.float32))
                scps.setdefault(od, []).append(f"{rid} {path.resolve()}\n")
                seconds += n / 16000
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
        logging.info(
            "%s %s look-ahead %d: %d recordings, %.2f h of audio in %d batches, "
            "%.0f s, peak GPU memory %.2f GiB",
            exp,
            ckpt,
            la,
            len(items),
            seconds / 3600,
            len(batches),
            time.perf_counter() - t0,
            peak,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    for od, lines in scps.items():
        (Path(od) / f"diarize.{task}.scp").write_text("".join(sorted(lines)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--out", required=True)
    pl.add_argument("--tasks", type=int, required=True)
    pl.add_argument(
        "--model", action="append", required=True, help="<name>=<checkpoint>"
    )
    pl.add_argument("--look_aheads", type=int, nargs="+", default=[0, 124])
    pl.add_argument("--sets", nargs="+", required=True)
    ru = sub.add_parser("run")
    ru.add_argument("plan")
    ru.add_argument("--batch_seconds", type=float, default=14400)
    a = ap.parse_args()
    cmd_plan(a) if a.cmd == "plan" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
