"""Cross-check the ST BLEU metric against egs2's own scoring pipeline.

`espnet3.systems.esp2_st.metrics.bleu.BLEU` exists to reproduce what
`egs2/TEMPLATE/st1/st.sh` reports (st.sh:1551-1620). Asserting that in prose is
not enough, so these tests run the shell pipeline itself -- the real
`utils/remove_punctuation.pl` and the `sacrebleu` CLI -- over shared fixtures
and require the numbers to agree exactly.

The fixtures in ``test_utils/espnet3/st`` are one reference and two hypothesis
sets, a close one and a poor one, so the comparison covers both a high and a
low score rather than a single operating point.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from espnet3.systems.esp2_st.metrics.bleu import BLEU

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = REPO_ROOT / "test_utils" / "espnet3" / "st"
REMOVE_PUNCTUATION_PL = REPO_ROOT / "utils" / "remove_punctuation.pl"

try:
    import sacrebleu  # noqa: F401

    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

needs_pipeline = pytest.mark.skipif(
    not HAS_SACREBLEU
    or shutil.which("perl") is None
    or shutil.which("sacrebleu") is None
    or not REMOVE_PUNCTUATION_PL.is_file(),
    reason="needs sacrebleu (CLI and module), perl, and utils/remove_punctuation.pl",
)


def _lines(name):
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


def _write_scp(path: Path, lines) -> Path:
    path.write_text(
        "\n".join(f"utt{i} {line}" for i, line in enumerate(lines)), encoding="utf-8"
    )
    return path


def _st_sh_scores(ref_path: Path, hyp_path: Path, lowercase: bool):
    """Score exactly as st.sh does, returning (bleu, chrf, ter).

    st.sh:1604 scores the detokenized text case-sensitively; st.sh:1609-1616
    pipes both sides through remove_punctuation.pl and scores again with -lc.
    Only BLEU takes -lc, which is sacreBLEU's own CLI behaviour
    (``dest='bleu_lowercase'``).
    """
    if lowercase:
        ref_path = _remove_punctuation(ref_path)
        hyp_path = _remove_punctuation(hyp_path)
    cmd = ["sacrebleu", str(ref_path), "-i", str(hyp_path), "-m", "bleu", "chrf", "ter"]
    if lowercase:
        cmd.append("-lc")
    cmd += ["-b", "-w", "2"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    scores = [float(x) for x in re.findall(r"-?\d+\.\d+", out)]
    assert len(scores) == 3, f"expected bleu/chrf/ter, parsed {scores} from {out!r}"
    return scores


def _remove_punctuation(path: Path) -> Path:
    out_path = path.with_suffix(path.suffix + ".lc.rm")
    with path.open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        subprocess.run(
            ["perl", str(REMOVE_PUNCTUATION_PL)], stdin=src, stdout=dst, check=True
        )
    return out_path


@needs_pipeline
@pytest.mark.parametrize("hyp_name", ["hyp_good.txt", "hyp_poor.txt"])
@pytest.mark.execution_timeout(120.0)
def test_matches_st_sh_pipeline(tmp_path: Path, hyp_name):
    """Every number this metric reports must equal what st.sh would print."""
    refs, hyps = _lines("ref.txt"), _lines(hyp_name)

    # The metric's own path: SCP files keyed by utterance id.
    result = BLEU(tokenize="13a")(
        {
            "ref": _write_scp(tmp_path / "ref.scp", refs),
            "hyp": _write_scp(tmp_path / "hyp.scp", hyps),
        },
        "fixture",
        tmp_path,
    )

    # The shell path: plain one-sentence-per-line files, as st.sh writes.
    plain_ref = tmp_path / "ref.trn.detok"
    plain_hyp = tmp_path / "hyp.trn.detok"
    plain_ref.write_text("\n".join(refs) + "\n", encoding="utf-8")
    plain_hyp.write_text("\n".join(hyps) + "\n", encoding="utf-8")

    bleu, chrf, ter = _st_sh_scores(plain_ref, plain_hyp, lowercase=False)
    assert result["BLEU"] == pytest.approx(bleu, abs=0.01)
    assert result["chrF2"] == pytest.approx(chrf, abs=0.01)
    assert result["TER"] == pytest.approx(ter, abs=0.01)

    bleu_lc, chrf_lc, ter_lc = _st_sh_scores(plain_ref, plain_hyp, lowercase=True)
    assert result["BLEU_lc"] == pytest.approx(bleu_lc, abs=0.01)
    assert result["chrF2_lc"] == pytest.approx(chrf_lc, abs=0.01)
    assert result["TER_lc"] == pytest.approx(ter_lc, abs=0.01)


@needs_pipeline
@pytest.mark.execution_timeout(120.0)
def test_ranks_the_two_hypothesis_sets_the_same_way_as_st_sh(tmp_path: Path):
    """A metric can match on absolute values and still order systems wrongly."""
    refs = _lines("ref.txt")
    scores = {}
    for name in ("hyp_good.txt", "hyp_poor.txt"):
        hyps = _lines(name)
        out = BLEU(tokenize="13a", chrf=False, ter=False)(
            {
                "ref": _write_scp(tmp_path / f"{name}.ref.scp", refs),
                "hyp": _write_scp(tmp_path / f"{name}.hyp.scp", hyps),
            },
            name,
            tmp_path,
        )
        scores[name] = out

    assert scores["hyp_good.txt"]["BLEU"] > scores["hyp_poor.txt"]["BLEU"]
    # The poor hypotheses are lowercased and lightly punctuated, so removing
    # case and punctuation should close much of the gap but not invert it.
    assert scores["hyp_good.txt"]["BLEU_lc"] > scores["hyp_poor.txt"]["BLEU_lc"]


@needs_pipeline
@pytest.mark.execution_timeout(120.0)
def test_lc_pass_really_strips_punctuation(tmp_path: Path):
    """Guard the specific bug this metric was fixed for.

    Lowercasing alone is NOT st.sh's case-insensitive score; it also runs
    remove_punctuation.pl. Scoring with that step disabled must differ, or the
    step is silently doing nothing.
    """
    refs, hyps = _lines("ref.txt"), _lines("hyp_poor.txt")
    data = {
        "ref": _write_scp(tmp_path / "ref.scp", refs),
        "hyp": _write_scp(tmp_path / "hyp.scp", hyps),
    }

    with_strip = BLEU(chrf=False, ter=False, lc_remove_punctuation=True)(
        data, "a", tmp_path
    )
    without_strip = BLEU(chrf=False, ter=False, lc_remove_punctuation=False)(
        data, "b", tmp_path
    )

    assert with_strip["BLEU_lc"] != without_strip["BLEU_lc"]
    # Only the stripped variant is what st.sh calls case-insensitive BLEU.
    plain_ref = tmp_path / "ref.trn.detok"
    plain_hyp = tmp_path / "hyp.trn.detok"
    plain_ref.write_text("\n".join(refs) + "\n", encoding="utf-8")
    plain_hyp.write_text("\n".join(hyps) + "\n", encoding="utf-8")
    expected, _, _ = _st_sh_scores(plain_ref, plain_hyp, lowercase=True)
    assert with_strip["BLEU_lc"] == pytest.approx(expected, abs=0.01)
