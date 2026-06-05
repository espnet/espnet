"""Smoke fine-tune + DER eval on AMI SDM (Array1-01 = array1 mic1).

Starts from the converted NVIDIA pretrained weights, fine-tunes the offline
Sortformer on a subset of AMI single-distant-mic (far-field) 30s windows, and
prints frame-level DER on a held-out SDM dev subset every few steps so the DER
trend is visible. This is a smoke/demonstration run, not a full training.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from egs3.librispeech_sortformer.diar.dataset import Dataset  # noqa: E402
from espnet2.diar.sortformer.convert_hf_sortformer import convert  # noqa: E402
from espnet2.tasks.diar_sortformer import SortformerDiarizationTask  # noqa: E402

AMI = "/raid/users/popcornell/AMI"
TRAIN_CUTS = f"{AMI}/ami-sdm_cuts_train_30s.jsonl.gz"
DEV_CUTS = f"{AMI}/ami-sdm_cuts_dev_30s.jsonl.gz"

N_TRAIN = 800
N_EVAL = 60
BATCH = 8
STEPS = 300
EVAL_EVERY = 30
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
# Full fine-tune of the pretrained model, adapting it to the far-field
# single-distant-mic (Array1-01) domain. DER should decrease from the
# pretrained baseline as the model adapts.
REINIT_DECODER = False


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
    print(f"device={DEVICE}  train_cuts={TRAIN_CUTS}")
    print("Loading + converting pretrained weights ...", flush=True)
    model, rep = convert("nvidia/diar_sortformer_4spk-v1", num_spk=4)
    print(f"  loaded {rep['n_loaded']} tensors", flush=True)

    if REINIT_DECODER:
        # Freeze the pretrained FastConformer feature extractor + frontend.
        for p in model.preprocessor.parameters():
            p.requires_grad_(False)
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        # Reinitialize the Transformer encoder + diarization head from scratch.
        from espnet2.diar.sortformer.sortformer_modules import SortformerModules
        from espnet2.diar.sortformer.transformer_encoder import TransformerEncoder

        model.transformer_encoder = TransformerEncoder(
            num_layers=18,
            hidden_size=192,
            inner_size=768,
            num_attention_heads=8,
            attn_score_dropout=0.5,
            attn_layer_dropout=0.5,
            ffn_dropout=0.5,
        )
        model.sortformer_modules = SortformerModules(
            num_spks=4, dropout_rate=0.5, fc_d_model=512, tf_d_model=192
        )
        print(
            "  REINIT: training fresh Transformer + head on FROZEN FastConformer "
            "features (far-field adaptation).",
            flush=True,
        )
    model.to(DEVICE)

    train_ds = Dataset(split="train", cuts=TRAIN_CUTS)
    eval_ds = Dataset(split="dev", cuts=DEV_CUTS)
    rng = np.random.RandomState(SEED)
    train_idx = rng.permutation(len(train_ds))[:N_TRAIN]
    eval_idx = rng.permutation(len(eval_ds))[:N_EVAL]
    print(f"train pool={len(train_idx)}  eval pool={len(eval_idx)}", flush=True)

    import argparse

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
        f"[step   0] DER(SDM/array1-mic1 dev) = {der0:.2f}%   (pretrained, pre-finetune)",
        flush=True,
    )

    ptr = 0
    order = list(train_idx)
    rng.shuffle(order)
    for step in range(1, STEPS + 1):
        model.train()
        if ptr + BATCH > len(order):
            rng.shuffle(order)
            ptr = 0
        batch_idx = order[ptr : ptr + BATCH]
        ptr += BATCH
        items = []
        for i in batch_idx:
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
        _, batch = collate(items)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss, stats, _ = model(**batch)
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % EVAL_EVERY == 0:
            der = evaluate(model, eval_ds, eval_idx)
            print(
                f"[step {step:3d}] loss={loss.item():.4f} "
                f"ats={float(stats['ats_loss']):.4f} pil={float(stats['pil_loss']):.4f} "
                f"f1={float(stats['f1_acc']):.3f}  ->  DER = {der:.2f}%",
                flush=True,
            )

    print("done.", flush=True)


if __name__ == "__main__":
    main()
