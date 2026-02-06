# Self-supervised Learning

This is a template of the ssl1 recipe for ESPnet2, designed for general purpose SSL.

## Differences from other recipes

ESPnet2 serves two different recipes for Self-Supervised Learning (SSL): `ssl1` (this one) and `hubert1`.

`hubert1` is the original implementation of SSL under the [HuBERT](https://arxiv.org/abs/2106.07447) pre-training framework. The recipe takes care of everything need for pre-training, such as K-means pseudo-labelling and discrete token evaluation. This is very important for reproducibility. However, it is quite complicated due to the multiple offline stages required for HuBERT and therefore difficult to hack/adapt to new training methods or other scenarios.

We created the new `ssl1` recipe to future-proof the codebase to accomodate other pre-training techniques that are purely end-to-end, such as [DinoSR](https://arxiv.org/abs/2305.10005), [SpeechFlow](https://arxiv.org/abs/2310.16338), or [w2v-BERT](https://arxiv.org/abs/2108.06209). This recipe is designed to be easily customizable and more scalable to large-scale pre-training setups.

## HuBERT Pre-training in SSL1

The `ssl1` codebase also supports HuBERT pre-training, but the steps to create the pseudo-labels are not included in the recipe. Users will either need to run the `hubert1` recipe to obtain the labels, or generate it themselves.

To use the labels from `hubert1`, follow these steps

0. Given a training set called `train_ssl` and a dev set called `dev_ssl`

1. Run `hubert1/hubert.sh` from stages 1 to 5 for a single iteration. This will generate:

    1. A token vocabulary list. It will be called something like `hubert1/data/en_token_list_kmeans_iter1_espnet_hubert_500clusters/word/tokens.txt`. The name will depend on your exact hyperparameters.

    2. A pseudo-label text file for both sets. The exact path will depend on your hyperparameters, but will look something like `hubert1/dump/<feat_type>/espnet_hubert/layer_<x>/<data split name>/pseudo_labels_km<num>.txt`.

2. Copy each `pseudo_labels_km<num>.txt` to the respective kaldi directly in `ssl` as `text`. For example: `cp hubert1/dump/ssl_feats/espnet_hubert/layer_9/train_ssl/pseudo_labels_km500.txt ssl1/dump/train_ssl/text`

3. In `ssl1/run.sh`, add the following flags:

    1. `--token_type word`

    2. `--token_list <path to token list from step 1.1>`

4. Update your training config with the used k-means size
    ```
    loss:
        - name: hubert
        conf:
        num_classes: <update this>
    ```
