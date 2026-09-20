# SPLET — Spoken Language Evaluation Toolkit

SPLET evaluates the **outputs** of spoken language systems: text and structured
text. Audio goes to [VERSA](https://github.com/wavlab-speech/versa); SPLET is
its text-side counterpart.

| Output type | Unified evaluation |
| --- | --- |
| Audio / speech / music | VERSA |
| Text / structured text | SPLET |

This is a **skeleton**. It fixes the package layout, the interface and the
scoring contract, and it implements exactly one metric — word/character error
rate — to pin that contract down. Everything else in espnet/espnet#6760 is
still to be written, and the directories where it goes say what is expected of
it.

## Status

| | |
| --- | --- |
| Implemented | WER, CER; the normalization pipeline; the utterance tier; `splet-score` |
| Contract only | the session tier (`splet/session_metrics`), the corpus tier (`splet/corpus_metrics`) |
| Not started | cpWER, ORC-WER, DER, JER, BLEU/chrF/TER, Whisper normalization, structured prediction |
| Not validated | the S/D/I split against sclite; everything against existing recipe scores |

WER and CER agree with `jiwer` on the corpus figure, which is what espnet3's
current metrics use (`test/splet/test_error_rate.py::test_matches_jiwer`).
Nothing here has been checked against SCTK yet, so do not report an S/D/I
breakdown from SPLET as SCTK-compatible.

## Running it

```bash
splet-score \
    --pred hyp.txt \
    --gt ref.txt \
    --score_config splet/egs/asr.yaml \
    --output_file result.jsonl
```

`--hyp` and `--ref` are accepted as aliases for `--pred` and `--gt`. The
canonical names are VERSA's, so that a recipe scores its audio and its text
with the same shape of command.

`--io` chooses how the two files are read: `kaldi` (a Kaldi `text` file),
`jsonl` (one JSON object per line, with `turns` for speaker-attributed or
timestamped output), or `dir`. Per-utterance results are written to
`--output_file` as JSON lines; the corpus summary is printed.

## The interface

A score config is a YAML list of metrics, exactly as in VERSA, optionally
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
def wer_metric(scorer, pred, gt) -> dict: ...  # score one item, flat dict out
```

Register it in `METRIC_CHOICES` in `splet/scorer_shared.py` with its tier.
VERSA dispatches with a long `if config["name"] == ...` chain instead; the
config, the function signatures and the output are the same either way.

## Three tiers

| Tier | Scored over | VERSA's equivalent |
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
than a factor of two on three utterances.

**Compatibility with an existing implementation is the requirement, not a
nice-to-have.** Every metric here has an implementation that recipes and
published results already use — sclite, sacrebleu, meeteval, dscore. A
reimplementation that is merely reasonable is worse than useless: it produces
numbers nobody can compare to anything. Each new metric lands with a test
against the tool it replaces, and any unavoidable difference gets written down.

## Why it is in the ESPnet repository

SPLET is meant to be extracted into its own repository and PyPI distribution
once the design settles, the way VERSA is separate today. Until then it lives
here, where the recipes and the reference scores it has to match also live.

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
`load_score_modules` / `list_scoring` / `load_summary` -- rather than SPLET
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
