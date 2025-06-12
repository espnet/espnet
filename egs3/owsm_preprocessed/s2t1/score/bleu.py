import re
from pathlib import Path
from typing import Dict, List, Union

from sacrebleu.metrics import BLEU as sacreBLEU
from sacrebleu.metrics import CHRF, TER

from espnet3.inference.abs_metrics import AbsMetrics


class BLEU(AbsMetrics):
    """
    Compute BLEU, CHRF, TER using sacrebleu, optionally with XML tag removal.

    Args:
        remove_xml (bool): If True, removes XML-like tags using regex.
    """

    def __init__(self, inputs, clean_types=None, remove_xml: bool = True):
        self.ref_key, self.hyp_key = inputs
        self.bleu = sacreBLEU()
        self.chrf = CHRF()
        self.ter = TER()
        self.remove_xml = remove_xml

    def clean(self, text: str) -> str:
        if self.remove_xml:
            text = re.sub(r"<[^>]+>", "", text)
        return text.strip() or "."

    def __call__(self, data: Dict[str, List[str]], test_name: str, decode_dir: Path) -> Dict[str, float]:
        refs = [self.clean(x) for x in data[self.ref_key]]
        hyps = [self.clean(x) for x in data[self.hyp_key]]

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

