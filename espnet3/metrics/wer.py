from pathlib import Path
from typing import Dict, List, Union

import jiwer

from espnet2.text.cleaner import TextCleaner
from espnet3.metrics.abs_metrics import AbsMetrics, validate_scp_files


class WER(AbsMetrics):
    """
    Compute WER using jiwer with ESPnet-style text cleaning.

    Args:
        clean_type (str): TextCleaner type, e.g., 'whisper_basic'.
    """

    def __init__(
        self, inputs: Union[List[str], Dict[str, str]], clean_types: str = None
    ):
        self.cleaner = TextCleaner(clean_types)
        self.inputs = inputs
        if isinstance(inputs, list):
            self.ref_key, self.hyp_key = inputs
        else:
            self.ref_key, self.hyp_key = "ref", "hyp"

    def clean(self, text: str) -> str:
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def __call__(self, decode_dir: Path, test_name: str) -> Dict[str, float]:
        scp_data = validate_scp_files(decode_dir, test_name, self.inputs)

        refs, hyps = [], []

        for uid in scp_data[self.ref_key]:
            ref = self.clean(scp_data[self.ref_key][uid])
            hyp = self.clean(scp_data[self.hyp_key][uid])
            refs.append(ref)
            hyps.append(hyp)

        wer_score = jiwer.wer(refs, hyps) * 100
        details = jiwer.process_words(refs, hyps)

        out_dir = decode_dir / test_name
        with open(out_dir / "wer_alignment", "w") as f:
            f.write(jiwer.visualize_alignment(details))

        return {
            "WER": round(wer_score, 2),
            "Substitutions": details.substitutions,
            "Insertions": details.insertions,
            "Deletions": details.deletions,
            "Hits": details.hits,
        }
