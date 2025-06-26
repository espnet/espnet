# README

This recipe provides a spoken language identification (LID) setup using the VoxLingua107 dataset, which contains over 6600 hours of speech in 107 languages. The speech segments are extracted from YouTube videos and labeled based on metadata, with a validated development set covering 33 languages.



## Results

**Overall Accuracy**

| Config                    | Accuracy (%) |
| ------------------------- | ------------ |
| `conf/mms_ecapa_baseline` | 94.3         |

**Per-Language Accuracy**

<div style="overflow-x: auto;">

| Config                    | ara  | aze  | cmn   | dan  | deu  | ell  | eng  | est  | fas  | fin  | fra  | hrv  | hun   | hye  | isl   | ita  | jpn  | lav   | lit  | mkd   | nld  | nno   | nor  | pol   | por   | rus  | slv  | spa  | srp  | swe  | tur  | ukr   | urd  |
| ------------------------- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ----- | ---- | ---- | ----- | ---- | ----- | ---- | ----- | ---- | ----- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- |
| `conf/mms_ecapa_baseline` | 94.0 | 94.1 | 100.0 | 97.0 | 93.9 | 80.0 | 88.8 | 95.3 | 92.0 | 96.8 | 97.0 | 50.0 | 100.0 | 96.0 | 100.0 | 98.0 | 97.6 | 100.0 | 84.6 | 100.0 | 95.0 | 100.0 | 70.8 | 100.0 | 100.0 | 93.1 | 88.9 | 92.7 | 85.7 | 97.0 | 95.8 | 100.0 | 78.9 |

</div>

**Note:**

The recommended transformers version is 4.51.3, and s3prl is 0.4.17.