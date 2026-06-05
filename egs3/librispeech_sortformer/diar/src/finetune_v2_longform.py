"""Per-window fine-tune of the streaming v2 model, evaluated on long-form DER.

Tests whether ordinary (offline, per-30s-window) fine-tuning of the streaming
Sortformer improves *long-form* (full-session, speaker-cache) DER on AMI SDM
(Array1-01 = array1 mic1). Reports collar-0.25 DER before and after.
"""

import argparse

import lhotse
import numpy as np
import torch

from egs3.librispeech_sortformer.diar.dataset import Dataset
from espnet2.diar.sortformer.convert_nemo_sortformer import convert_nemo
from espnet2.diar.sortformer.longform import run_longform_inference, score_rttm_der
from espnet2.tasks.diar_sortformer import SortformerDiarizationTask

AMI = "/raid/users/popcornell/AMI"
TRAIN_CUTS = f"{AMI}/ami-sdm_cuts_train_30s.jsonl.gz"
V2 = "/tmp/sortformer_v2_full.pt"
N_TRAIN = 6000
BATCH = 8
STEPS = 500
LR = 2e-5
WARMUP = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def longform_dev_der(model, tag):
    rec = lhotse.load_manifest(f"{AMI}/ami-sdm_recordings_dev.jsonl.gz")
    sup = lhotse.load_manifest(f"{AMI}/ami-sdm_supervisions_dev.jsonl.gz")
    cuts = lhotse.CutSet.from_manifests(recordings=rec, supervisions=sup)
    out = f"/tmp/lf_v2_{tag}"
    run_longform_inference(model, cuts, out, mode="streaming", device=DEVICE, log=None)
    res = score_rttm_der(out, collar=0.25)
    print(f"[{tag}] long-form AMI SDM dev DER: {res}", flush=True)
    return res


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device={DEVICE} per-window fine-tune of streaming v2", flush=True)
    model, _ = convert_nemo(V2, num_spk=4)
    model.to(DEVICE)

    longform_dev_der(model, "baseline")  # before fine-tuning

    train_ds = Dataset(split="train", cuts=TRAIN_CUTS)
    rng = np.random.RandomState(SEED)
    order = list(rng.permutation(len(train_ds))[:N_TRAIN])
    collate = SortformerDiarizationTask.build_collate_fn(argparse.Namespace(), train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    model.train()
    ptr = 0
    for step in range(1, STEPS + 1):
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
        if step % 100 == 0:
            print(
                f"[step {step:3d}] loss={loss.item():.4f} f1={float(stats['f1_acc']):.3f}",
                flush=True,
            )

    torch.save(model.state_dict(), "/tmp/sortformer_v2_ami_ft.pth")
    longform_dev_der(model, "finetuned")  # after fine-tuning


if __name__ == "__main__":
    main()
