# Self-supervised Learning

This is a template of the hubert11 recipe for ESPnet2, designed for HuBERT-style SSL.

## Differences from other recipes

ESPnet2 serves two different recipes for Self-Supervised Learning (SSL): `ssl1` and `hubert1` (this one).

`hubert1` is the original implementation of SSL under the [HuBERT](https://arxiv.org/abs/2106.07447) pre-training framework. The recipe takes care of everything need for pre-training, such as K-means pseudo-labelling and discrete token evaluation. This is very important for reproducibility. However, it is quite complicated due to the multiple offline stages required for HuBERT and therefore difficult to hack/adapt to new training methods or other scenarios.

We created the new `ssl1` recipe to future-proof the codebase to accomodate other pre-training techniques that are purely end-to-end, such as [DinoSR](https://arxiv.org/abs/2305.10005), [SpeechFlow](https://arxiv.org/abs/2310.16338), or [w2v-BERT](https://arxiv.org/abs/2108.06209). This recipe is designed to be easily customizable and more scalable to large-scale pre-training setups.

Note: the `ssl1` codebase also supports HuBERT pre-training, but the steps to create the pseudo-labels are not included in the recipe. Users will either need to run the `hubert1` recipe to obtain the labels, or generate it themselves.
