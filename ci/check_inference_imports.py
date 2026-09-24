#!/usr/bin/env python3
"""Import every inference entry point with espnet installed bare.

`pip install espnet` with no extras is the install a user gets from the README
quick start, a Hugging Face Space's requirements.txt, or `uvx espnet-mcp`, and
it must be enough to load a pretrained model and run it. The training stack
(lightning, wandb, tensorboard, torch_optimizer, matplotlib, nltk, and hydra,
omegaconf, datasets, dask for espnet3) lives in the [train] extra, so this
script fails the moment a module on the inference path starts importing one of
them at module level. test_import_all.py, run with [all] installed, covers
everything else.

Some entry points need a task extra by design - speechlm's inference imports
transformers, duckdb, lhotse and liger_kernel at module level - and
`--extra NAME` checks those after `pip install "espnet[NAME]"` in the same
bare environment, which is what catches an extra that lost a package.
"""

import importlib
import sys
import traceback

MODULES = [
    # the top-level package: `import espnet; espnet.load(tag)` is the entry
    # point a user reaches for first, and it ships in the same wheel
    "espnet",
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
    "espnet2.bin.cli",
    # gradio is in [demo], not in the bare install: this module must import
    # without it, or `espnet demo` would be the one command that cannot even
    # report what is missing.
    "espnet2.bin.demo",
    "espnet2.bin.mcp_server",
]

# Entry points that need one task extra installed on top of the bare package.
EXTRA_MODULES = {
    "speechlm": ["espnet2.speechlm.bin.inference"],
}

extras = sys.argv[1:]
if extras and extras[0] == "--extra":
    modules, what = EXTRA_MODULES[extras[1]], f"[{extras[1]}]"
else:
    modules, what = MODULES, "a bare install"

failed = []
for name in modules:
    print(f"import {name}", file=sys.stderr)
    try:
        importlib.import_module(name)
    except Exception:
        failed.append((name, traceback.format_exc()))

if failed:
    print(f"Error: {len(failed)} inference modules do not import with {what}")
    for i, (name, reason) in enumerate(failed, 1):
        print(f"[{i}] {name}\n\t{reason}\n")
    raise SystemExit(1)
print(f"OK: {len(modules)} inference modules import with {what}")
