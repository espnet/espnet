# URGENT24 audio quality assessment

Train Uni-VERSA to predict precomputed evaluation metrics for URGENT24 audio.
This recipe consumes externally prepared audio and scores. It does not download
the corpus or compute the target metrics. See the [AQA task guide](../../TEMPLATE/aqa1/README.md)
for the label format and shared stages.

Prepare `data/train`, `data/dev`, and `data/test`. Each split must contain
`wav.scp`, `ref_wav.scp`, `text`, `utt2spk`, and `metric.scp`, keyed by the same
utterance IDs. Audio entries use Kaldi SCP format; `ref_wav.scp` may use
`utt_id None` for a missing reference. Text entries contain the reference
transcript. Each `metric.scp` entry contains an utterance ID and a JSON object
of numeric metric values, for example:

```text
utt1 {"pesq": 2.1, "wer": 10.0}
```

Omit unavailable metric labels from the JSON object. `local/data.sh` checks
that the files exist, sorts them, and generates `spk2utt` from `utt2spk`.
Stage 3 discovers metric names from the training labels.

From this directory, run the baseline with log-mel features:

```bash
./run.sh
```

To use WavLM Large features, install S3PRL with the ESPnet tools installer
(`cd ../../../tools && make s3prl.done`), then run:

```bash
./run.sh --train_config conf/tuning/train_wavlm.yaml
```

The WavLM frontend downloads pretrained weights on first use. Both configurations
use 16 kHz audio and reference text/audio. The default is one training GPU;
use `--ngpu 0` for CPU training. Override shared driver options through `run.sh`,
for example `--stage 6 --stop_stage 7` to collect statistics and train after data
preparation. These are corpus training configurations, not CI debug settings.
