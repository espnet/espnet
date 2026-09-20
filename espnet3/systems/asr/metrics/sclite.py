"""ESPnet2-compatible word, character and subword error rates via SCTK."""

import json
import re
import shutil
import subprocess
from pathlib import Path

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric


class ScliteErrorRate(BaseMetric):
    """Use the tokenizer and SCTK scoring rules from ESPnet2's asr.sh.

    This optional metric preserves SCTK's case handling and edit alignment,
    which can differ from JiWER. Install SCTK with tools/installers/install_sctk.sh
    and put tools/sctk/bin on PATH before running the measure stage.

    Args:
        token_type: word, char, or bpe, selecting WER, CER, or TER.
        bpemodel: SentencePiece model path, required for bpe.
        clean_types: Reference cleaners; defaults to no cleaning.
        hyp_clean_types: Hypothesis cleaners; independently defaults to none.
        non_linguistic_symbols: Optional symbol file, as in asr.sh.
        sclite: Executable name on PATH or an absolute executable path.
        ref_key: Reference SCP input alias.
        hyp_key: Hypothesis SCP input alias.

    Raises:
        ValueError: Unsupported token type or a missing bpe model setting.
    """

    def __init__(
        self,
        token_type="word",
        bpemodel=None,
        clean_types=None,
        hyp_clean_types=None,
        non_linguistic_symbols=None,
        sclite="sclite",
        ref_key="ref",
        hyp_key="hyp",
    ):
        """Build the same tokenizers and separate cleaners as stage 13."""
        if token_type not in ("word", "char", "bpe"):
            raise ValueError(f"Unsupported scoring token_type: {token_type}")
        if token_type == "bpe" and bpemodel is None:
            raise ValueError("bpe scoring requires bpemodel")
        self.name = {"word": "WER", "char": "CER", "bpe": "TER"}[token_type]
        self.tokenizer = build_tokenizer(
            token_type=token_type,
            bpemodel=bpemodel,
            non_linguistic_symbols=non_linguistic_symbols,
            remove_non_linguistic_symbols=token_type in ("word", "char"),
        )
        self.ref_cleaner = TextCleaner(clean_types)
        self.hyp_cleaner = TextCleaner(hyp_clean_types)
        self.sclite = sclite
        self.ref_key, self.hyp_key = ref_key, hyp_key

    def __call__(self, data, test_name, inference_dir):
        """Write TRN inputs, run native SCTK, and return the aggregate error rate.

        Args:
            data: Mapping of ref_key and hyp_key to aligned SCP files.
            test_name: Dataset output subdirectory.
            inference_dir: Root for score_<metric> artifacts. Repeated calls
                replace reports but do not modify inference inputs.

        Returns:
            A mapping such as {"WER": 12.34}, in percent rounded to two digits.
            Exact integer counts are also written to counts.json. As in SCTK,
            an empty reference denominator is reported as zero percent.

        Raises:
            RuntimeError: SCTK is unavailable or its total-count row is absent.
            subprocess.CalledProcessError: The original SCTK command fails.
            AssertionError: Reference and hypothesis SCP IDs are not aligned.
        """
        executable = shutil.which(self.sclite)
        if executable is None:
            raise RuntimeError("SCTK sclite is required; add tools/sctk/bin to PATH")
        output = Path(inference_dir) / test_name / f"score_{self.name.lower()}"
        output.mkdir(parents=True, exist_ok=True)
        ref_path, hyp_path = output / "ref.trn", output / "hyp.trn"
        with ref_path.open("w") as refs, hyp_path.open("w") as hyps:
            for index, (_, row) in enumerate(
                self.iter_inputs(data, self.ref_key, self.hyp_key)
            ):
                for stream, key, cleaner in (
                    (refs, self.ref_key, self.ref_cleaner),
                    (hyps, self.hyp_key, self.hyp_cleaner),
                ):
                    tokens = self.tokenizer.text2tokens(
                        cleaner(" ".join(row[key].split()))
                    )
                    # IDs are only alignment keys; totals are independent of
                    # speaker grouping. Preserve empty predictions verbatim.
                    stream.write(f"{' '.join(tokens)} (spk-{index})\n")
        with (output / "result.txt").open("w") as report:
            subprocess.run(
                [
                    executable,
                    "-r",
                    str(ref_path),
                    "trn",
                    "-h",
                    str(hyp_path),
                    "trn",
                    "-i",
                    "rm",
                    "-o",
                    "all",
                    "stdout",
                ],
                stdout=report,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report = (output / "result.txt").read_text()
        match = re.search(r"\|\s*Sum\s*\|([^\n]+)", report)
        if match is None:
            raise RuntimeError("SCTK report lacks the raw Sum row")
        values = [int(value) for value in re.findall(r"\d+", match.group(1))]
        counts = dict(
            zip(
                (
                    "sentences",
                    "reference_tokens",
                    "correct",
                    "substitutions",
                    "deletions",
                    "insertions",
                    "errors",
                    "sentence_errors",
                ),
                values,
                strict=True,
            )
        )
        (output / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
        denominator = counts["reference_tokens"]
        score = 100 * counts["errors"] / denominator if denominator else 0.0
        return {self.name: round(score, 2)}
