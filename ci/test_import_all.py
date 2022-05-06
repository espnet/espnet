#!/usr/bin/env python3
import glob
import importlib
import sys

try:
    import k2
except Exception:
    has_k2 = False
else:
    has_k2 = True
try:
    import mir_eval
except Exception:
    has_mir_eval = False
else:
    has_mir_eval = True


for dirname in ["espnet", "espnet2"]:
    for f in glob.glob(f"{dirname}/**/*.py"):
        module_name = f.replace("/", ".")[:-3]

        if (
            (
                not has_k2
                and (
                    module_name == "espnet2.bin.asr_inference_k2"
                    or module_name == "espnet2.fst.lm_rescore"
                )
            )
            or (
                not has_mir_eval
                and (
                    module_name == "espnet2.bin.enh_scoring"
                    or module_name == "espnet2.bin.enh_scoring_flexible_numspk"
                )
            )
            or module_name == "espnet2.tasks.enh_asr"
        ):
            print(f"[Skip] import {module_name}", file=sys.stderr)
            continue
        else:
            print(f"import {module_name}", file=sys.stderr)

        importlib.import_module(module_name)
