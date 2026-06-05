"""Smoke test: 8-spk streaming Sortformer, NEST-init, local 20 s window, 1-min
context, trained on FastMSS + AMI; long-form DER on AMI dev (valid) + test.

Exercises the full main-recipe pipeline end to end on a small scale:
  1. generate a few FastMSS simulated meetings (3-8 spk, 60 s),
  2. build AMI SDM 1-min train windows, combine with FastMSS,
  3. build the 8-spk model (FastConformer = NEST-L, efficient O(N*W) sliding
     window ~20 s; Transformer also windowed; speaker cache; train_streaming),
     load ONLY the NEST encoder weights,
  4. train a few hundred steps,
  5. long-form streaming DER on AMI dev + test (before vs after).

Run (NEST weights at /tmp/nest_encoder.pt):
  CUDA_VISIBLE_DEVICES=2 python -m egs3.librispeech_sortformer.diar.src.smoke_fastmss_ami_local
"""

import argparse

import lhotse
import numpy as np
import torch
from lhotse.manipulation import combine as combine_cuts

from egs3.librispeech_sortformer.diar.dataset import Dataset
from egs3.librispeech_sortformer.diar.src.data_prep import build_ami_cuts, run_fastmss
from espnet2.diar.espnet_sortformer_model import ESPnetSortformerModel
from espnet2.diar.sortformer.convert_nest import load_nest_encoder
from espnet2.diar.sortformer.fastconformer_encoder import FastConformerEncoder
from espnet2.diar.sortformer.longform import run_longform_inference, score_rttm_der
from espnet2.diar.sortformer.preprocessor import MelSpectrogramPreprocessor
from espnet2.diar.sortformer.sortformer_modules import SortformerModules
from espnet2.diar.sortformer.transformer_encoder import TransformerEncoder
from espnet2.tasks.diar_sortformer import SortformerDiarizationTask

AMI = "/raid/users/popcornell/AMI"
NEST = "/tmp/nest_encoder.pt"
FASTMSS_DIR = "/tmp/FastMSS_ls"
ALIGNED = "/raid/users/popcornell/sim_data/librispeech_aligned_manifests"
DATA = "/tmp/smoke_local_data"
FASTMSS_OUT = "/tmp/smoke_local_fastmss"

NUM_SPK = 8
# 80 ms frames after 8x subsampling -> ~12.5 fps. 20 s window ~= 250 frames span.
ATT = [125, 125]  # local window ~20 s (+/-10 s)
CHUNK_LEN = 750  # ~1 min context per streaming chunk
SPKCACHE_LEN = 188  # ~15 s speaker cache
WINDOW = 60.0  # 1-min training sessions
N_FASTMSS = 60
N_TRAIN = 400
BATCH = 2
STEPS = 300
LR = 1e-4
WARMUP = 50
DROPOUT = 0.1  # lower than released 0.5 for a small-data smoke
N_DEV = 4
N_TEST = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def build_model():
    pre = MelSpectrogramPreprocessor(
        sample_rate=16000, features=80, normalize="per_feature"
    )
    enc = FastConformerEncoder(
        feat_in=80,
        d_model=512,
        n_layers=18,
        n_heads=8,
        ff_expansion_factor=4,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        conv_kernel_size=9,
        dropout=0.1,
        dropout_att=0.1,
        att_context_size=ATT,
    )
    mods = SortformerModules(
        num_spks=NUM_SPK,
        fc_d_model=512,
        tf_d_model=192,
        dropout_rate=DROPOUT,
        spkcache_len=SPKCACHE_LEN,
        fifo_len=0,
        chunk_len=CHUNK_LEN,
        spkcache_update_period=SPKCACHE_LEN,
    )
    tf = TransformerEncoder(
        num_layers=18,
        hidden_size=192,
        inner_size=768,
        num_attention_heads=8,
        attn_score_dropout=DROPOUT,
        attn_layer_dropout=DROPOUT,
        ffn_dropout=DROPOUT,
        att_context_size=ATT,
    )
    model = ESPnetSortformerModel(
        pre, enc, mods, tf, num_spk=NUM_SPK, train_streaming=True
    )
    load_nest_encoder(model, NEST)
    for p in model.preprocessor.parameters():
        p.requires_grad_(False)
    return model


def eval_longform(model, split, n, tag):
    rec = lhotse.load_manifest(f"{AMI}/ami-sdm_recordings_{split}.jsonl.gz")
    sup = lhotse.load_manifest(f"{AMI}/ami-sdm_supervisions_{split}.jsonl.gz")
    cuts = lhotse.CutSet.from_manifests(recordings=rec, supervisions=sup)
    cuts = cuts.subset(first=n)
    out = f"/tmp/smoke_local_lf_{split}_{tag}"
    run_longform_inference(model, cuts, out, mode="streaming", device=DEVICE)
    der = score_rttm_der(out, collar=0.25)
    print(
        f"[{tag}] AMI {split} long-form DER = {der.get('DER'):.2f}%  ({der})",
        flush=True,
    )
    return der


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    print(f"device={DEVICE} num_spk={NUM_SPK} att={ATT} chunk={CHUNK_LEN}", flush=True)

    # 1) FastMSS meetings (reuse if present)
    synth = f"{FASTMSS_OUT}/manifests/synth-librispeech-train-cuts.jsonl.gz"
    import os

    if not os.path.exists(synth):
        synth = run_fastmss(
            fastmss_dir=FASTMSS_DIR,
            output_dir=FASTMSS_OUT,
            librispeech_dir="/raid/users/popcornell/LibriSpeech",
            librispeech_align="unused",
            aligned_manifests=ALIGNED,
            noise_folders=None,
            n_meetings=N_FASTMSS,
            min_max_spk=(3, 8),
            duration=60,
            dset_splits=("train-clean-100",),
            reverberate=True,
            n_jobs=8,
        )

    # 2) AMI SDM 1-min train windows + combine with FastMSS
    ami = build_ami_cuts(AMI, DATA, cond="sdm", window=WINDOW, splits=("train",))
    fast = lhotse.load_manifest(synth).cut_into_windows(WINDOW)
    fast = fast.filter(lambda c: len(c.supervisions) > 0).to_eager()
    amitr = lhotse.load_manifest(ami["train"])
    comb = combine_cuts([fast, amitr])
    comb_path = f"{DATA}/train_combined_cuts.jsonl.gz"
    comb.to_file(comb_path)
    print(
        f"train cuts: FastMSS={len(fast)} + AMI={len(amitr)} = {len(comb)}", flush=True
    )

    # 3) model
    model = build_model().to(DEVICE)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"trainable params = {ntrain:.1f}M (NEST encoder loaded)", flush=True)

    train_ds = Dataset(split="train", cuts=comb_path, num_spk=NUM_SPK)
    order = list(rng.permutation(len(train_ds))[:N_TRAIN])
    collate = SortformerDiarizationTask.build_collate_fn(
        argparse.Namespace(), train=True
    )

    # 4) DER before training (chance baseline) on a couple dev sessions
    eval_longform(model, "dev", 2, "before")

    # 5) train
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-3
    )
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
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, step / max(1, WARMUP))
        opt.step()
        if step % 25 == 0:
            print(
                f"[step {step:3d}] loss={loss.item():.4f} "
                f"f1={float(stats['f1_acc']):.3f}",
                flush=True,
            )

    # 6) long-form DER after training on AMI dev (valid) + test
    eval_longform(model, "dev", N_DEV, "after")
    eval_longform(model, "test", N_TEST, "after")
    print("smoke done.", flush=True)


if __name__ == "__main__":
    main()
