"""From-scratch Sortformer training on AMI SDM (Array1-01), NEST-initialized.

Loads ONLY the NEST self-supervised weights into the FastConformer encoder
(`nvidia/ssl_en_nest_large_v1.0`, the way NVIDIA initializes Sortformer); the
Transformer encoder and diarization head start from random init. Trains the
whole model on AMI single-distant-mic windows and prints frame-level DER on a
held-out dev subset so the DER descent (from chance) is visible.

Export the NEST encoder weights first (NeMo env):
    from nemo.collections.asr.models import ASRModel
    m = ASRModel.from_pretrained("nvidia/ssl_en_nest_large_v1.0", map_location="cpu")
    import torch; torch.save(m.encoder.state_dict(), "/tmp/nest_encoder.pt")
"""

import argparse

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from egs3.librispeech_sortformer.diar.dataset import Dataset  # noqa: E402
from espnet2.diar.espnet_sortformer_model import build_sortformer_model  # noqa: E402
from espnet2.diar.sortformer.convert_nest import load_nest_encoder  # noqa: E402
from espnet2.tasks.diar_sortformer import SortformerDiarizationTask  # noqa: E402

AMI = "/raid/users/popcornell/AMI"
TRAIN_CUTS = f"{AMI}/ami-sdm_cuts_train_30s.jsonl.gz"
DEV_CUTS = f"{AMI}/ami-sdm_cuts_dev_30s.jsonl.gz"
NEST = "/tmp/nest_encoder.pt"

N_TRAIN = 1600
N_EVAL = 60
BATCH = 8
STEPS = 320
EVAL_EVERY = 40
LR = 5e-5
WARMUP = 40
# Lower decoder dropout than the released 0.5: from a random init on a small real
# set, 0.5 collapses to a degenerate constant. (The released config's 0.5 assumes
# training on massive simulated data.)
DECODER_DROPOUT = 0.1
FREEZE_ENCODER = False  # NVIDIA: full fine-tune (no freezing) works best
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def frame_der(ref, hyp, thres=0.5):
    hyp = (hyp >= thres).astype(np.float32)
    t = min(ref.shape[0], hyp.shape[0])
    ref, hyp = ref[:t], hyp[:t]
    s = max(ref.shape[1], hyp.shape[1])
    ref = np.pad(ref, ((0, 0), (0, s - ref.shape[1])))
    hyp = np.pad(hyp, ((0, 0), (0, s - hyp.shape[1])))
    cost = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            cost[i, j] = np.sum(np.abs(ref[:, i] - hyp[:, j]))
    ri, ci = linear_sum_assignment(cost)
    hp = np.zeros_like(ref)
    for i, j in zip(ri, ci):
        hp[:, i] = hyp[:, j]
    miss = np.sum((ref == 1) & (hp == 0))
    fa = np.sum((ref == 0) & (hp == 1))
    return float(ref.sum()), float(miss), float(fa)


@torch.no_grad()
def evaluate(model, eval_ds, idxs):
    model.eval()
    tot_ref = tot_err = 0.0
    for i in idxs:
        d = eval_ds[i]
        wav = torch.tensor(d["speech"]).unsqueeze(0).to(DEVICE)
        ln = torch.tensor([wav.shape[1]]).to(DEVICE)
        preds, plen = model.diarize(wav, ln)
        preds = preds[0, : plen[0]].float().cpu().numpy()
        r, miss, fa = frame_der(d["spk_labels"], preds)
        tot_ref += r
        tot_err += miss + fa
    return 100.0 * tot_err / max(tot_ref, 1.0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(
        f"device={DEVICE}  init=NEST(only)  freeze_encoder={FREEZE_ENCODER}", flush=True
    )
    model = build_sortformer_model(  # random init everywhere ...
        num_spk=4,
        tf_dropout=DECODER_DROPOUT,
        sortformer_dropout=DECODER_DROPOUT,
    )
    load_nest_encoder(model, NEST)  # ... then load ONLY the NEST encoder weights
    if FREEZE_ENCODER:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    for p in model.preprocessor.parameters():
        p.requires_grad_(False)
    model.to(DEVICE)

    train_ds = Dataset(split="train", cuts=TRAIN_CUTS)
    eval_ds = Dataset(split="dev", cuts=DEV_CUTS)
    rng = np.random.RandomState(SEED)
    train_idx = list(rng.permutation(len(train_ds))[:N_TRAIN])
    eval_idx = list(rng.permutation(len(eval_ds))[:N_EVAL])

    collate = SortformerDiarizationTask.build_collate_fn(
        argparse.Namespace(), train=True
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"trainable params = {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True
    )
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-3)

    der0 = evaluate(model, eval_ds, eval_idx)
    print(
        f"[step   0] DER(SDM/array1-mic1 dev) = {der0:.2f}%   (random head, NEST encoder)",
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
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        for g in opt.param_groups:  # linear LR warmup
            g["lr"] = LR * min(1.0, step / max(1, WARMUP))
        opt.step()
        if step % EVAL_EVERY == 0:
            der = evaluate(model, eval_ds, eval_idx)
            print(
                f"[step {step:3d}] loss={loss.item():.4f} f1={float(stats['f1_acc']):.3f}"
                f"  ->  DER = {der:.2f}%",
                flush=True,
            )
    print("done.", flush=True)


if __name__ == "__main__":
    main()
