# WHAM! snoar recipe

Label-free speech enhancement on WHAM! (16 kHz, single speaker, `min`) with
the ESPnet3 SNOAR system (`espnet3/systems/snoar`):

- **DNF** (Maciejewski et al., arXiv:2606.02327): train `x1 + n2 -> (x1, n2)`
  on noisy recordings `x1` and noise-only signals `n2`; enhance with the
  eq.-14 projection.
- **MixIT** (Wisdom et al., arXiv:2006.12701): constrained 3-slot MixIT on the
  same `(x1, n2)`, with energy mixture consistency.
- **SNOAR**: self-training of a seed from the noisy recordings alone. An EMA
  teacher relabels each recording; the student learns from the
  observation-added target `(1 - gamma) s_hat + gamma x1` remixed with another
  recording's noise estimate at a self-estimated SNR, with a two-view
  consistency term. `gamma` is cosine-annealed 0.2 -> 0.05 over 16
  generations.
- **RemixIT** (Tzinis et al., arXiv:2202.08862) from the same seed, as the
  baseline SNOAR is compared with.

Seeds can be label-free (DNF, MixIT) or supervised out of domain (the FastMSS
Fisher model below, or any ESPnet2 enhancement model such as
`kohei0209/tfgridnet_urgent25`). Only SNOAR/RemixIT read the WHAM! recordings
without any noise-only signal; DNF/MixIT need `n2`, which comes either from an
external noise corpus or from the speech-free gaps of the recordings
(`harvest_noise`, from VAD labels or an energy VAD).

The same system backs the `espnet snoar` command line, which adapts a model to
any folder of noisy recordings:

```bash
espnet snoar adapt my_recordings/ -o my_enhancer/          # from the URGENT 2025 seed
espnet snoar adapt my_recordings/ -o my_enhancer/ --seed dnf --vad labels.rttm
espnet snoar enhance noisy.wav -o clean.wav --model my_enhancer/
```

## Data Preparation

WHAM! needs the LDC WSJ0 corpus and cannot be downloaded by the recipe.
Generate the 16 kHz `min` files with the WHAM!/WHAMR! scripts
(<https://wham.whisper.ai>; the WHAMR! tree `wav16k/min` also holds the
anechoic WHAM! files) and point `WHAM_ROOT` at the directory holding
`tr/ cv/ tt/`:

```bash
export WHAM_ROOT=/path/to/whamr/wav16k/min
export WHAM_OUTPUT=$PWD/data          # where the manifests go (optional)
```

`create_dataset` writes JSONL manifests (one file per split, no per-utterance
files):

| manifest | content |
|---|---|
| `train.jsonl` | half of `tr` (stable hash of the utterance id): the noisy recordings `x1`, no references |
| `noise_wham.jsonl` | the noise files of the other half of `tr`: the `n2` pool for DNF/MixIT, at native level |
| `valid.jsonl`, `test.jsonl` | `cv`, `tt` with `speech_ref` (validation diagnostic and scoring only) |

`n1` and `n2` therefore never share a clip, as in the reference. Every mode
trains on the same `x1` half.

The supervised Fisher seeds read a float16 cache of the FastMSS Fisher
meetings (LDC data, keep it private); set `FISHER_CACHE` to it. Build it once
from the FastMSS Shar shards:

```bash
python -m espnet3.systems.snoar.src.data.fisher <fastmss_shar_dir> $FISHER_CACHE \
  --kinds speech dry noise --jobs 11
```

`speech` is the sum of the reverberant speaker images and `dry` the
sample-aligned anechoic speech. The speech target is always the speech exactly
as it is in the input: the **reverberant** image for reverberant crops, the
dry speech for anechoic ones.

`training_supervised_fisher.yaml` is the paper TF-GridNet with two outputs
(speech, noise) and energy mixture consistency, trained on 50 % reverberant +
noise and 50 % anechoic + noise crops at -8..10 dB SNR. Its noise is BFND
only (`noise_bfnd.txt`, one path per line, hash-split into train/valid),
because the FastMSS noise track was drawn partly from the `high_res_wham`
recordings that WHAM! `tt` noise is cut from. It is read with the DNF
projection (`readout: dnf`). A one-output seed read as `raw` has no way to
remove noise it has learned to keep, so under observation adding SNOAR drifts
to the identity (measured: URGENT25 seed 12.8 dB -> -1.2 dB SI-SDR on `tt`).

## Commands

Run from `egs3/wham/snoar` after `. ./path.sh`.

```bash
# manifests
python run.py --stages create_dataset --training_config conf/training_dnf.yaml

# label-free seeds (one GPU each)
python run.py --stages train export --training_config conf/training_dnf.yaml
python run.py --stages train export --training_config conf/training_mixit.yaml
# DNF with no external noise: pool = energy-VAD gaps of the recordings
python run.py --stages harvest_noise train export --training_config conf/training_dnf_vad.yaml
# supervised out-of-domain seeds (FastMSS Fisher)
python run.py --stages train export --training_config conf/training_supervised_fisher.yaml
python run.py --stages train export --training_config conf/training_supervised_fisher_bsrnn.yaml

# SNOAR self-training from a seed's export (and the RemixIT baseline)
python run.py --stages train export --training_config conf/training_snoar_dnf.yaml
python run.py --stages train export --training_config conf/training_snoar_mixit.yaml
python run.py --stages train export --training_config conf/training_snoar_supervised_fisher.yaml
python run.py --stages train export --training_config conf/training_snoar_supervised_fisher_1view.yaml
python run.py --stages train export --training_config conf/training_snoar_supervised_fisher_bsrnn.yaml
python run.py --stages train export --training_config conf/training_remixit_dnf.yaml

# enhance tt and score SI-SDR / PESQ (wb) / STOI; set exp_tag in both configs
python run.py --stages infer measure \
  --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

`export` writes `exp/<tag>/export/` (`config.yaml` + `model.pth`, an ordinary
ESPnet2 enhancement model, and `snoar.yaml` with the readout). It is the seed
of the next run, the model `infer` loads, and what
`espnet snoar enhance --model` takes.

Validation reports `valid/si_sdr` against the `cv` references for
monitoring. It does not select checkpoints: `export` reads `last.ckpt`, as
the reference does, so no label of the adaptation data is used.

## Results

WHAM! `tt` (3000 utterances, 16 kHz, `min`), scored by `measure` with
`conf/inference.yaml` and `conf/metrics.yaml` (`exp_tag` set to the run).
Every model is read from `export/` (`last.ckpt`, no checkpoint selection on
labels). The supervised seeds are out of domain: they never see WHAM! audio
or WHAM! noise. These numbers come from this port, not from the reference
implementation of the SNOAR paper (bDNF, commit 2b1e186).

### Supervised Fisher seeds and SNOAR self-training

| Model | Config | SI-SDR (dB) | PESQ-WB | STOI |
|---|---|---|---|---|
| TF-GridNet seed (150k steps) | `training_supervised_fisher.yaml` | 6.87 | 1.49 | 0.847 |
| SNOAR from it, one view | `training_snoar_supervised_fisher_1view.yaml` | 12.61 | 1.95 | 0.931 |
| BSRNN-64 seed (50k steps) | `training_supervised_fisher_bsrnn.yaml` | 7.92 | 1.53 | 0.850 |
| SNOAR from it, one view | `training_snoar_supervised_fisher_bsrnn.yaml` | 11.81 | 1.67 | 0.916 |

The 4x longer BSRNN SNOAR schedule
(`training_snoar_supervised_fisher_bsrnn_long.yaml`) was stopped at
generation 13 with validation SI-SDR flat at the level of the short one.
The two-view TF-GridNet run (`training_snoar_supervised_fisher.yaml`) was
stopped early, because it gained little over one view at about twice the cost;
it has no `tt` result.

The DNF, MixIT, RemixIT and DNF-VAD configs have only been run as short
checks so far; no full-length result is reported for them. No pretrained
model of this recipe is published: the Fisher seeds are trained on LDC data
and FSD50K clips of mixed licenses. A commercially usable seed trained on
URGENT 2025 data is in `egs3/urgent25/snoar`.

### Differences from the reference implementation

The training step is the reference's `bdnf_cyc` arm: same losses, OA blend,
two-view consistency, self-SNR-matched donors, teacher readouts, projection
guard, EMA decay, Adam with cosine LR decay to 5 %, and gradient clipping at 5.
The known differences are:

- The self-estimated SNR bank is filled from the first training batch onwards;
  the reference warms it with 64 teacher passes before step 0.
- Inference output is gain-aligned to the input (`align_scale`), and long
  recordings are processed in 20 s chunks with 50 % Hann overlap-add. Neither
  changes SI-SDR, STOI or PESQ, which are scale-invariant or level-normalized.
- The CLI rescales an external noise corpus to 0-10 dB below the recording;
  the recipe, like the reference, uses WHAM! noise at its native level.
