#!/usr/bin/env python3
import glob
import importlib

try:
    import k2
except Exception:
    has_k2 = False
else:
    has_k2 = True


for dirname in ["espnet", "espnet2"]:
    for f in glob.glob(f"{dirname}/**/*.py"):
        module_name = f.replace("/", ".")[:-3]

        if (
            not has_k2 and module_name == "espnet2.bin.k2_asr_inference"
        ) or module_name == "espnet2.tasks.enh_asr":
            print(f"[Skip] import {module_name}")
            continue
        else:
            print(f"import {module_name}")

        importlib.import_module(module_name)
