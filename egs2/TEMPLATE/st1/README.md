# ESPnet-ST-v2: Speech-to-Text Translation

TODO

## Table of Contents
* Recipe Flow
* Offline ST Models
   * Results
   * Core Architectures
      * Attentional Encoder-Decoder
      * CTC/Attention
      * Transducer
      * Multi-Decoder
   * Auxiliary Techniques
      * Pre-training
      * SSL Front-end/Encoder
      * LLM Decoder
      * Hierarchical Encoding
      * Minimum Bayes-Risk Decoding
* Streaming ST Models
   * Results
   * Core Architectures
      * Blockwise CTC/Attention
      * Label-Synchronous Blockwise CTC/Attention
      * Time-Synchronous Blockwise CTC/Attention
      * Blockwise Transducer
   * Auxiliary Techniques
      * Using an offline model for streaming inference
* FAQ
* How to cite

## Recipe Flow

TODO

## Offline ST Models

TODO

### Results

|Model|BLEU|Model Link|Training Config|Decoding Config|
|---|---|---|---|---|
|Attentional Encoder-Decoder|25.7|link|link|link|
|Multi-Decoder Attn Enc-Dec|27.6|link|link|link|
|CTC/Attention|28.6|link|link|link|
|Multi-Decoder CTC/Attention|28.8|link|link|link|
|Transducer|27.6|link|link|link|

## Streaming ST Models

TODO

### Results

|Model|BLEU|AL|Model Link|Training Config|Decoding Config|
|---|---|---|---|---|---|
|Blockwise Attn Enc-Dec|22.8|3.23|link|link|link|
|Label-Sync Blockwise CTC/Attn|24.4|3.23|link|link|link|
|Time-Sync Blockwise CTC/Attn|24.6|2.34|link|link|link|
|Blockwise Transducer|22.9|2.37|link|link|link|

