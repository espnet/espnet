"""Fine-tune the pretrained Sortformer (full model) on AMI SDM and save it.

Adapts NVIDIA's converted weights to the far-field single-distant-mic domain by
training on AMI SDM 30 s windows, saving the resulting checkpoint for long-form
evaluation. A cheap per-window (chunk) DER on a dev subset is printed during
training as a progress signal; the comparable long-form DER is computed
afterwards with ``eval_longform_ami.py`` + ``score_der_pyannote.py``.
"""

import argparse

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from egs3.librispeech_sortformer.diar.dataset import Dataset  # noqa: E402
from espnet2.diar.sortformer.convert_hf_sortformer import convert  # noqa: E402
from espnet2.tasks.diar_sortformer import SortformerDiarizationTask  # noqa: E402

AMI = "/raid/users/popcornell/AMI"
TRAIN_CUTS = f"{AMI}/ami-sdm_cuts_train_30s.jsonl.gz"
DEV_CUTS = f"{AMI}/ami-sdm_cuts_dev_30s.jsonl.gz"

N_TRAIN = 6000
N_EVAL = 60
BATCH = 8
STEPS = 700
EVAL_EVERY = 100
LR = 2e-5
WARMUP = 50
SAVE = "/tmp/sortformer_ami_sdm_ft.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def chunk_der(model, ds, idxs):
    model.eval()
    tot_ref = tot_err = 0.0
    with torch.no_grad():
        for i in idxs:
            d = ds[i]
            wav = torch.tensor(d["speech"]).unsqueeze(0).to(DEVICE)
            ln = torch.tensor([wav.shape[1]]).to(DEVICE)
            preds, plen = model.diarize(wav, ln)
            hyp = (preds[0, : plen[0]].float().cpu().numpy() >= 0.5).astype(np.float32)
            ref = d["spk_labels"]
            t = min(ref.shape[0], hyp.shape[0])
            ref, hyp = ref[:t], hyp[:t]
            s = max(ref.shape[1], hyp.shape[1])
            ref = np.pad(ref, ((0, 0), (0, s - ref.shape[1])))
            hyp = np.pad(hyp, ((0, 0), (0, s - hyp.shape[1])))
            cost = np.array(
                [
                    [np.sum(np.abs(ref[:, i] - hyp[:, j])) for j in range(s)]
                    for i in range(s)
                ]
            )
            ri, cj = linear_sum_assignment(cost)
            hp = np.zeros_like(ref)
            for a, b in zip(ri, cj):
                hp[:, a] = hyp[:, b]
            tot_ref += float(ref.sum())
            tot_err += float(
                np.sum((ref == 1) & (hp == 0)) + np.sum((ref == 0) & (hp == 1))
            )
    return 100.0 * tot_err / max(tot_ref, 1.0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device={DEVICE} full fine-tune from pretrained -> {SAVE}", flush=True)
    model, _ = convert("nvidia/diar_sortformer_4spk-v1", num_spk=4)
    model.to(DEVICE)

    train_ds = Dataset(split="train", cuts=TRAIN_CUTS)
    eval_ds = Dataset(split="dev", cuts=DEV_CUTS)
    rng = np.random.RandomState(SEED)
    train_idx = list(rng.permutation(len(train_ds))[:N_TRAIN])
    eval_idx = list(rng.permutation(len(eval_ds))[:N_EVAL])

    collate = SortformerDiarizationTask.build_collate_fn(
        argparse.Namespace(), train=True
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    print(
        f"[step   0] chunk-DER(dev) = {chunk_der(model, eval_ds, eval_idx):.2f}%",
        flush=True,
    )
    order = list(train_idx)
    rng.shuffle(order)
    ptr = 0
    for step in range(1, STEPS + 1):
        model.train()
        if ptr + BATCH > len(order):
            rng.shuffle(order)
            ptr = 0
        items = []
        for i in order[ptr : ptr + BATCH]:
            d = train_ds[i]
            items.append(
                (
                    d["utt_id"],
                    {
                        "speech": d["speech"].astype(np.float32),
                        "spk_labels": d["spk_labels"].astype(np.float32),
                    },
                )
            )
        ptr += BATCH
        _, batch = collate(items)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss, stats, _ = model(**batch)
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, step / max(1, WARMUP))
        opt.step()
        if step % EVAL_EVERY == 0:
            print(
                f"[step {step:3d}] loss={loss.item():.4f} f1={float(stats['f1_acc']):.3f} "
                f"chunk-DER(dev) = {chunk_der(model, eval_ds, eval_idx):.2f}%",
                flush=True,
            )
    torch.save(model.state_dict(), SAVE)
    print(f"saved fine-tuned checkpoint -> {SAVE}", flush=True)


if __name__ == "__main__":
    main()
