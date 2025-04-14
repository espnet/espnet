from espnet3.metrics.abs_metrics import AbsMetrics, validate_scp_files
from espnet2.text.cleaner import TextCleaner
from pathlib import Path
from typing import List, Dict, Union
import jiwer


class CER(AbsMetrics):
    """
    Compute CER using jiwer with ESPnet-style text cleaning.
    
    Args:
        clean_type (str): TextCleaner type, e.g., 'whisper_basic'.
    """

    def __init__(self, clean_type: str = "whisper_basic"):
        self.cleaner = TextCleaner([clean_type])

    def clean(self, text: str) -> str:
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def __call__(self, decode_dir: Path, test_name: str, inputs: Union[List[str], Dict[str, str]]) -> Dict[str, float]:
        scp_data = validate_scp_files(decode_dir, test_name, inputs)
        if isinstance(inputs, list):
            ref_key, hyp_key = inputs
        else:
            ref_key, hyp_key = "ref", "hyp"

        refs, hyps = [], []

        for uid in scp_data[ref_key]:
            ref = self.clean(scp_data[ref_key][uid])
            hyp = self.clean(scp_data[hyp_key][uid])
            refs.append(ref)
            hyps.append(hyp)

        cer_score = jiwer.cer(refs, hyps) * 100
        details = jiwer.process_words(refs, hyps)

        out_dir = decode_dir / test_name
        with open(out_dir / "cer_alignment", "w") as f:
            f.write(jiwer.visualize_alignment(details))

        return {
            "CER": round(cer_score, 2),
            "Substitutions": details.substitutions,
            "Insertions": details.insertions,
            "Deletions": details.deletions,
            "Hits": details.hits
        }
