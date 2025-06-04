from pathlib import Path
from typing import Dict, List, Union

import jiwer

from espnet2.text.cleaner import TextCleaner
from espnet3.inference.abs_metrics import AbsMetrics


class CER(AbsMetrics):
    def __init__(self, inputs, clean_types=None):
        self.cleaner = TextCleaner(clean_types)
        self.inputs = inputs
        self.ref_key, self.hyp_key = inputs

    def clean(self, text: str) -> str:
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def __call__(self, data: Dict[str, List[str]], test_name: str, decode_dir: Path) -> Dict[str, float]:
        refs = [self.clean(x) for x in data[self.ref_key]]
        hyps = [self.clean(x) for x in data[self.hyp_key]]

        score = jiwer.cer(refs, hyps) * 100
        details = jiwer.process_words(refs, hyps)

        with open(decode_dir / test_name / "cer_alignment", "w") as f:
            f.write(jiwer.visualize_alignment(details))

        return {
            "CER": round(score, 2),
        }

