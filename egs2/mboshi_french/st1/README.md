# Recipe for Mboshi-French Speech translation corpus.

Our paper : https://arxiv.org/pdf/2204.02470.pdf

Dataset paper : https://arxiv.org/pdf/1710.03501.pdf

- lang : mdw , fra

### Baseline Transformer ([pretrained model](https://huggingface.co/espnet/speech_translation_mboshi_french_transformer_baseline/tree/main))
|dataset|BLEU|chrF2|TER|
|---|---|---|---|
|dev|10.9|24.5|87.5|

### Linear-based Fusion 
|dataset|BLEU|chrF2|TER|
|---|---|---|---|
|dev|11.6|24.6|81.4|

### Convolution-based Fusion 
|dataset|BLEU|chrF2|TER|
|---|---|---|---|
|dev|11.3|24.7|83.9|

### Coattention-based Fusion
|dataset|BLEU|chrF2|TER|
|---|---|---|---|
|dev|10.9|24.8|85.9|

### MixtureOfExperts-based Fusion
|dataset|BLEU|chrF2|TER|
|---|---|---|---|
|dev|11.2|24.7|83.9|
