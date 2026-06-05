"""NeMo-faithful training of the 8-spk streaming Sortformer for AMI long-form DER.

Matches NeMo's released streaming-v2 config (`diar_streaming_sortformer_4spk-v2`
model_config.yaml): session_len 90 s, chunk_len/spkcache_len/update_period 188,
fifo_len 0, spkcache_sil_frames_per_spk 3, FULL attention (att_context [-1,-1]),
AdamW lr 1e-4 betas (0.9,0.98) wd 1e-3 + InverseSquareRootAnnealing warmup 500,
batch 4. `causal_attn_rate` is skipped on purpose (no causal/low-latency
streaming). The speaker **cache is in the loop** (train_streaming) so it learns
to read/compress the cache. Intended departures from v2: 8 speakers (v2 = 4) and
NEST-init (80-mel / 18-layer; v2 = 128-mel / 17-layer). Data: FastMSS (3-8 spk,
reverb + WHAM noise) + AMI SDM, all windowed to 90 s.

Reports AMI dev long-form DER during training (best-checkpointed) + final
dev/test DER. Single-GPU.

  CUDA_VISIBLE_DEVICES=2 python -m egs3.librispeech_sortformer.diar.src.train_scaled_local
"""

import argparse
import os

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
NOISE = "/raid/users/popcornell/whamr/wav16k/min/tr/noise"  # WHAM noise (looped)
DATA = "/raid/users/popcornell/sortformer_scaled_exp/data"
FASTMSS_OUT = "/raid/users/popcornell/sortformer_scaled_exp/fastmss"
EXP = "/raid/users/popcornell/sortformer_scaled_exp"

NUM_SPK = 8
# NeMo-faithful: full attention (att_context_size=[-1,-1]); causal_attn_rate=0
# (we are NOT doing causal/low-latency streaming, so skip that augmentation).
ATT = None
CHUNK_LEN = 188  # NeMo v2: chunk_len 188 (~15 s)
SPKCACHE_LEN = 188  # NeMo v2: spkcache_len 188
SIL_FRAMES = 3  # NeMo v2: spkcache_sil_frames_per_spk 3
WINDOW = 90.0  # NeMo v2: session_len_sec 90
N_FASTMSS = 1500
BATCH = 4  # NeMo v2: batch_size 4 (fits at 90 s: ~24 GB)
STEPS = 6000
LR = 1e-4
WARMUP = 500  # NeMo v2: InverseSquareRootAnnealing warmup_steps 500
DROPOUT = 0.1
EVAL_AT = [1500, 3000, 4500, 6000]
N_DEV_EVAL = 5  # periodic dev sessions
N_DEV_FINAL = 8
N_TEST_FINAL = 8
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
        spkcache_sil_frames_per_spk=SIL_FRAMES,
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
    cuts = lhotse.CutSet.from_manifests(recordings=rec, supervisions=sup).subset(
        first=n
    )
    out = f"{EXP}/lf_{split}_{tag}"
    run_longform_inference(model, cuts, out, mode="streaming", device=DEVICE)
    der = score_rttm_der(out, collar=0.25)
    print(
        f"[{tag}] AMI {split} long-form DER = {der.get('DER'):.2f}%  {der}", flush=True
    )
    return der.get("DER", 1e9)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    os.makedirs(EXP, exist_ok=True)
    print(
        f"device={DEVICE} num_spk={NUM_SPK} att={ATT} chunk={CHUNK_LEN} "
        f"cache={SPKCACHE_LEN} batch={BATCH} steps={STEPS}",
        flush=True,
    )

    # 1) FastMSS (reverb + WHAM noise), reuse if present
    synth = f"{FASTMSS_OUT}/manifests/synth-librispeech-train-cuts.jsonl.gz"
    if not os.path.exists(synth):
        synth = run_fastmss(
            fastmss_dir=FASTMSS_DIR,
            output_dir=FASTMSS_OUT,
            librispeech_dir="/raid/users/popcornell/LibriSpeech",
            librispeech_align="unused",
            aligned_manifests=ALIGNED,
            noise_folders=[NOISE],
            n_meetings=N_FASTMSS,
            min_max_spk=(3, 8),
            duration=120,
            dset_splits=("train-clean-100", "train-clean-360"),
            reverberate=True,
            n_jobs=16,
            extra_overrides=["filter_noise_len=2"],
        )

    # 2) combine FastMSS (windowed) + AMI SDM 1-min train
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
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"trainable params = {nt:.1f}M (NEST loaded)", flush=True)
    train_ds = Dataset(split="train", cuts=comb_path, num_spk=NUM_SPK)
    order = list(rng.permutation(len(train_ds)))
    collate = SortformerDiarizationTask.build_collate_fn(
        argparse.Namespace(), train=True
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.98),
        weight_decay=1e-3,
    )

    def lr_at(step):  # NeMo InverseSquareRootAnnealing
        if step < WARMUP:
            return LR * step / WARMUP
        return LR * (WARMUP / step) ** 0.5

    best_der, best_path = 1e9, f"{EXP}/best.pt"
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
            g["lr"] = lr_at(step)
        opt.step()
        if step % 50 == 0:
            print(
                f"[step {step:4d}] loss={loss.item():.4f} f1={float(stats['f1_acc']):.3f}"
                f" lr={lr_at(step):.2e}",
                flush=True,
            )
        if step in EVAL_AT:
            der = eval_longform(model, "dev", N_DEV_EVAL, f"step{step}")
            if der < best_der:
                best_der = der
                torch.save(model.state_dict(), best_path)
                print(
                    f"  ** new best dev DER {best_der:.2f}% -> {best_path}", flush=True
                )
            model.train()

    # 4) final eval with best checkpoint
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"loaded best (dev DER {best_der:.2f}%) for final eval", flush=True)
    eval_longform(model, "dev", N_DEV_FINAL, "final_dev")
    eval_longform(model, "test", N_TEST_FINAL, "final_test")
    print("scaled training done.", flush=True)


if __name__ == "__main__":
    main()
