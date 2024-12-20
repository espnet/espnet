# ML-SUPERB 2.0 2024 Challenge


This is a recipe to reproduce the baseline model for the [Interspeech 2024 ML-SUPERB 2.0 Challenge](multilingual.superbbenchmark.org). While the challenge is open-ended, the organizers have provided here a minimal training and development set based off of the [ML-SUPERB 2.0 Benchmark](https://www.isca-archive.org/interspeech_2024/shi24g_interspeech.pdf) for participants to use. This data will cover most of the evaluated languages. More information about the challenge and the dataset construction can be found on the [challenge website](https://multilingual.superbbenchmark.org/challenge-interspeech2025/challenge_overview).


The baseline uses frozen SSL features from [MMS 1B](https://www.jmlr.org/papers/v25/23-1318.html), which are input into a 2-layer Transformer trained using CTC loss. It takes roughly 2 days to train on a single H100 GPU.


The challenge will use a custom scoring script, which considers worst language performance and CER standard deviation in addition to the typical multilingual ASR metrics of language identification accuracy and ASR CER. The exact implementation can be found in `local/score.py`.
