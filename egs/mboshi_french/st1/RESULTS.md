### Conformer + ASR-MTL + MT-MTL + SpecAugment (200ep)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_nodev_sp.fr_lc_pytorch_train_pytorch_conformer_kernel15_half_specaug/decode_dev.fr_decode_pytorch_transformer|**9.74**|26.3|11.7|6.9|4.5|0.988|0.989|3962|4008|

### Conformer + ASR-MTL + MT-MTL (100ep)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_nodev_sp.fr_lc_pytorch_train_pytorch_conformer_kernel15_half_specaug/decode_dev.fr_decode_pytorch_transformer|8.50|27.4|11.3|5.8|3.4|0.960|0.960|3849|4008|

### Transformer + ASR-MTL + MT-MTL + SpecAugment (100ep)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_nodev_sp.fr_lc_pytorch_train_pytorch_transformer_specaug/decode_dev.fr_decode_pytorch_transformer|2.47|17.7|4.1|1.6|0.6|0.841|0.852|3415|4008|

### Transformer + ASR-MTL + MT-MTL (100ep)
|dataset|BLEU|1-gram|2-gram|3-gram|4-gram|BP|ratio|hyp_len|ref_len|
|---|---|---|---|---|---|---|---|---|---|
|exp/train_nodev_sp.fr_lc_pytorch_train_pytorch_transformer/decode_dev.fr_decode_pytorch_transformer|6.18|20.8|7.6|4.2|2.5|0.968|0.968|3881|4008|
