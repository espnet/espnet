Silent speech recognition using ultrasound image features. Modified basing on WSJ recipe.

The dateset is from SSC dataset(Silent Speech Challenge, https://ftp.espci.fr/pub/sigma/)

Instructions:
1. Replaced `stage 3: LM Preparation` with downloading WSJ pretrained LM. （See https://github.com/espnet/espnet/blob/master/egs/wsj/asr1/RESULTS.md). This is becauese, in SSC data, train set transcripts are from TIMIT corpus, but test set transcripts are from WSJ corpus.
2. Default features are 30 dimentions DCT feature.According to the paper [Updating the silent speech challenge benchmark with deep learning](https://arxiv.org/abs/1709.06818), DCT features performance is better than auto endcoder. Also you can try it by yourself. Just download features from the SSC dataset (https://ftp.espci.fr/pub/sigma/Features/) to the `data` directory, then replace the path in`test.scp` with your work path.
3. Actually this recipe uses parameters based on the Voxfage recipe, tuned `batchsize` 64 >> 16 and `transformer-lr` 10 >> 20.

Here's my decode results with `batch size` 12 and `transformer-lr` 20. https://storage.googleapis.com/szxs/batch12lr20.zip

I'll try SpecAug further.
