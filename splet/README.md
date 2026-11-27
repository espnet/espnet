# SPLET — Spoken Language Evaluation Toolkit

SPLET evaluates the **outputs** of spoken language systems: text and structured
text. Audio goes to [VERSA](https://github.com/wavlab-speech/versa); SPLET is
its text-side counterpart.

| Output type | Unified evaluation |
| --- | --- |
| Audio / speech / music | VERSA |
| Text / structured text | SPLET |

This is a **skeleton**. It fixes the package layout, the interface and the
metric contract, and implements word/character error rate in the utterance
tier and BLEU/chrF/TER (through sacrebleu) in the corpus tier. Everything else
in espnet/espnet#6760 is still to be written, and the directories where it goes
say what is expected of it.

## Status

| | |
| --- | --- |
| Implemented | WER, CER; BLEU, chrF, TER via sacrebleu; the normalization pipeline; the utterance and corpus tiers; `splet-measure` |
| Contract only | the session tier (`splet/session_metrics`) |
| Not started | cpWER, ORC-WER, DER, JER, multi-reference BLEU, Whisper normalization, structured prediction |
| Not validated | the S/D/I split against sclite; everything against existing recipe results |

WER and CER agree with `jiwer` on the corpus figure, which is what espnet3's
current metrics use (`test/splet/test_error_rate.py::test_matches_jiwer`).
Nothing here has been checked against SCTK yet, so do not report an S/D/I
breakdown from SPLET as SCTK-compatible.

BLEU, chrF and TER are sacrebleu's own numbers, and the summary's `metadata`
carries the signature sacrebleu reports for each. They match the `sacrebleu` commands that
`egs2/TEMPLATE/st1/st.sh` runs, with and without `-lc`
(`test/splet/test_bleu.py`); `splet/egs/st.yaml` is the case-sensitive
pass. The case-insensitive pass also strips punctuation with
`remove_punctuation.pl` first, which SPLET's `remove_punctuation` does not
match on non-ASCII symbols. st.sh's Moses tokenize/detokenize round trip is
not part of SPLET: measure detokenized text, as st.sh does.

## Running it

```bash
splet-measure \
    --pred hyp.txt \
    --gt ref.txt \
    --metrics_config splet/egs/asr.yaml \
    --output_file result.jsonl
```

`--hyp` and `--ref` are accepted as aliases for `--pred` and `--gt`. The
canonical names are VERSA's, so that a recipe measures its audio and its text
with the same shape of command.
The command itself is `splet-measure`, after espnet3's `measure` stage, not
`splet-score`: in ESPnet a *scorer* is a beam-search component, so SPLET
avoids the word.

`--io` chooses how the two files are read: `kaldi` (a Kaldi `text` file),
`jsonl` (one JSON object per line, with `turns` for speaker-attributed or
timestamped output), or `dir`. Per-utterance results are written to
`--output_file` as JSON lines; the corpus summary, including corpus-tier
metrics such as BLEU, is printed.

## The interface

A metrics config is a YAML list of metrics, exactly as in VERSA, optionally
wrapped in a mapping that adds a shared normalization pipeline:

```yaml
normalize:
  - name: lowercase
  - name: remove_punctuation
    keep: "'"
metrics:
  - name: wer
  - name: cer
    tokenizer_conf:
      remove_space: false
```

A metric is a pair of plain functions, which is VERSA's contract unchanged:

```python
def wer_setup(**kwargs) -> Any: ...            # build whatever state it needs
def wer_metric(state, pred, gt) -> dict: ...  # measure one item, flat dict out
```

Register it in `METRIC_CHOICES` in `splet/metric_registry.py` as a
`MetricSpec`, which declares four things nothing else should have to guess:

| Field | Says |
| --- | --- |
| `tier` | which loop runs it: `utterance`, `session`, `corpus` |
| `requires` | what it needs beyond the hypothesis: `reference`, `timestamps`, `speakers`; checked before anything is measured |
| `outputs` | every key it reports (by the suffix after its id) and how the summary reduces it: `sum` for a count, `rate` for `_errors / _ref_len`, `text` for text |
| `version` | the implementation's version, recorded with every result |

A config entry names the implementation (`name`) and may give the instance
its own `id`; the id prefixes every key the instance reports, so one
implementation can run twice in one config (`wer` normalized, `wer_raw` on
the raw text) without one overwriting the other. A repeated id is an error.

VERSA dispatches with a long `if config["name"] == ...` chain instead; the
config, the function signatures and the output are the same either way.

## Three tiers

| Tier | Measured over | VERSA's equivalent |
| --- | --- | --- |
| `utterance_metrics` | one hypothesis/reference pair | `utterance_metrics` |
| `session_metrics` | one recording | *(none)* |
| `corpus_metrics` | the whole corpus at once | `corpus_metrics` |

The session tier is the one VERSA does not have, and the reason it exists here
from the start: cpWER, ORC-WER, DER and JER are defined over a whole
recording — a speaker permutation only means anything within one — and are
then pooled across recordings. Long-form and multi-speaker evaluation is a
tier, not an extension to be bolted onto the utterance path later.

## Two rules that are not style preferences

**Error rates are pooled, not averaged.** A metric reports its counts next to
its rate (`wer`, `wer_errors`, `wer_ref_len`, `wer_sub`, `wer_del`, `wer_ins`),
and the summary recomputes `sum(errors) / sum(ref_len)`. That is what SCTK
reports, and it is not the mean of the per-utterance rates unless every
utterance is the same length — in `test_error_rate.py` the two differ by more
than a factor of two on three utterances. The summary applies the rule each
metric *declares* in its spec; a key no metric declared is an error, never a
guess. An empty reference follows one policy at both levels
(`error_rate_from_counts`): with no reference tokens, any error is a rate of
1.0 and no error is 0.0, and the counts are reported next to the rate so
nothing is hidden.

**Corpus figures are accumulated, so a training loop can use them.** The
summary is `Accumulator`: `add` folds one result's counts in by the declared
rules, `result` recomputes the pooled rates, `state` hands the counts out as
plain numbers for an `all_reduce` across workers, and `measure_batch` measures
two parallel lists without utterance ids. A validation step therefore adds one
batch and reads the corpus WER at the end of the epoch; `summarize` is the
same accumulator over a complete list. This is the shape the espnet3 metric
contract (espnet/espnet#6843) adapts to: `update` is `measure_batch` plus
`add`, `compute` is `result`, `reset` is `reset`.

**A result carries what produced it.** The printed summary ends with a
versioned `metadata` block (`splet/metadata.py`): per metric, the
implementation name and version, its tier and requirements, the resolved
configuration it was built from (normalization pipeline included), the
alignment backend, and the reference tool's signature where there is one.
A number without that block cannot be reproduced or compared.

**Compatibility with an existing implementation is the requirement, not a
nice-to-have.** Every metric here has an implementation that recipes and
published results already use — sclite, sacrebleu, meeteval, dscore. A
reimplementation that is merely reasonable is worse than useless: it produces
numbers nobody can compare to anything. Each new metric lands with a test
against the tool it replaces, and any unavoidable difference gets written down.

**Missing is an error; empty is a deletion.** A reference with no hypothesis,
or a hypothesis with no reference, stops the run with a `KeyError` naming the
utterance (`require_matching_keys`, which every tier calls first). Dropping
either side would change the denominator and improve the result without
anyone asking for it. A duplicate id in either input is an error for the same
reason: whichever line won, the other's words would vanish from the count.
A metric's declared requirements are checked first of all
(`validate_requirements`): WER without a reference fails naming the metric,
not half way through with a `TypeError`. An utterance the system produced nothing
for is not missing: it appears in the hypothesis file as its ID with an empty
text (the Kaldi reader accepts that line), and every reference word counts as
a deletion. The two cases are deliberately told apart.

## Why it is in the ESPnet repository

SPLET is meant to be extracted into its own repository and PyPI distribution
once the design settles, the way VERSA is separate today. Until then it lives
here, where the recipes and the reference results it has to match also live.

For that extraction to stay a directory move rather than a porting project,
nothing inside `splet/` may import ESPnet — which is exactly what the espnet3
metrics it will replace do today (`from espnet2.text.cleaner import
TextCleaner`). `ci/check_splet_independence.py` fails the build if that
changes, and also if `splet/` imports a third-party package the `splet` extra
does not declare. The dependency runs one way only:

```
espnet3  ──imports──>  splet
```

So an espnet3 metric becomes a thin adapter over SPLET -- it keeps the
`BaseMetric` interface, reads the SCP files espnet3 hands it, and calls
`load_metrics` / `measure_utterances` / `summarize` -- rather than SPLET
growing an understanding of espnet3. No adapter is written yet: replacing the
existing `espnet3/systems/asr/metrics` is the first milestone of
espnet/espnet#6760, and it can only happen after SPLET reproduces what those
metrics report today.

## Tests

```bash
pytest test/splet/
python3 ci/check_splet_independence.py
```

The suite is written to skip rather than fail when an optional package is
absent, so it also runs against a bare install. A CI job that installs *only*
the declared dependencies, and so catches an undeclared import the developer
machine hides, does not exist yet and should.
