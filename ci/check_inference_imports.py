#!/usr/bin/env python3
"""Import every inference entry point with espnet installed bare.

`pip install espnet` with no extras is the install a user gets from the README
quick start, a Hugging Face Space's requirements.txt, or `uvx espnet-mcp`, and
it must be enough to load a pretrained model and run it. The training stack
(lightning, wandb, tensorboard, torch_optimizer, matplotlib, nltk) and the
espnet3 stack (hydra, omegaconf, datasets, dask) live in extras, so this
script fails the moment a module on the inference path starts importing one of
them at module level. test_import_all.py, run with [all] installed, covers
everything else.
"""

import importlib
import sys
import traceback

MODULES = [
    "espnet2.bin.asr_inference",
    "espnet2.bin.asr_inference_streaming",
    "espnet2.bin.asr_inference_maskctc",
    "espnet2.bin.asr_transducer_inference",
    "espnet2.bin.s2t_inference",
    "espnet2.bin.s2t_inference_ctc",
    "espnet2.bin.s2t_inference_language",
    "espnet2.bin.tts_inference",
    "espnet2.bin.tts2_inference",
    "espnet2.bin.svs_inference",
    "espnet2.bin.enh_inference",
    "espnet2.bin.enh_inference_streaming",
    "espnet2.bin.enh_tse_inference",
    "espnet2.bin.st_inference",
    "espnet2.bin.st_inference_streaming",
    "espnet2.bin.mt_inference",
    "espnet2.bin.s2st_inference",
    "espnet2.bin.spk_inference",
    "espnet2.bin.lid_inference",
    "espnet2.bin.slu_inference",
    "espnet2.bin.lm_inference",
    "espnet2.bin.diar_inference",
    "espnet2.bin.cls_inference",
    "espnet2.bin.gan_codec_inference",
    "espnet2.bin.uasr_inference",
    "espnet2.bin.mcp_server",
]

failed = []
for name in MODULES:
    print(f"import {name}", file=sys.stderr)
    try:
        importlib.import_module(name)
    except Exception:
        failed.append((name, traceback.format_exc()))

if failed:
    print(f"Error: {len(failed)} inference modules do not import with a bare install")
    for i, (name, reason) in enumerate(failed, 1):
        print(f"[{i}] {name}\n\t{reason}\n")
    raise SystemExit(1)
print(f"OK: {len(MODULES)} inference modules import with a bare install")
