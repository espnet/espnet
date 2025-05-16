import re
from pathlib import Path
from typing import Dict, List, Union

from sacrebleu.metrics import BLEU as sacreBLEU
from sacrebleu.metrics import CHRF, TER

from espnet3.metrics.abs_metrics import AbsMetrics, validate_scp_files


class BLEU(AbsMetrics):
    """
    Compute BLEU, CHRF, TER using sacrebleu, optionally with XML tag removal.

    Args:
        remove_xml (bool): If True, removes XML-like tags using regex.
    """

    def __init__(self, remove_xml: bool = True):
        self.bleu = sacreBLEU()
        self.chrf = CHRF()
        self.ter = TER()
        self.remove_xml = remove_xml

    def clean(self, text: str) -> str:
        if self.remove_xml:
            text = re.sub(r"<[^>]+>", "", text)
        return text.strip() or "."

    def __call__(
        self, decode_dir: Path, test_name: str, inputs: Union[List[str], Dict[str, str]]
    ) -> Dict[str, Union[float, str]]:
        scp_data = validate_scp_files(decode_dir, test_name, inputs)
        if isinstance(inputs, list):
            ref_key, hyp_key = inputs
        else:
            ref_key, hyp_key = "ref", "hyp"

        refs = [self.clean(scp_data[ref_key][uid]) for uid in scp_data[ref_key]]
        hyps = [self.clean(scp_data[hyp_key][uid]) for uid in scp_data[hyp_key]]

        score_dir = decode_dir / test_name

        bleu = self.bleu.corpus_score(hyps, [refs])
        chrf = self.chrf.corpus_score(hyps, [refs])
        ter = self.ter.corpus_score(hyps, [refs])

        with open(score_dir / "score", "w") as f:
            f.write("\n".join([str(bleu), str(chrf), str(ter)]))

        return {
            "BLEU": round(bleu.score, 3),
            "CHRF": round(chrf.score, 3),
            "TER": round(ter.score, 3),
        }
