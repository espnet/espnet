# MELD speech emotion recognition

Upload speech or record with the microphone to classify the emotion of an
utterance as one of `neutral`, `joy`, `surprise`, `anger`, `sadness`,
`disgust` or `fear`.

The model is a frozen [WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm)
frontend with a Transformer encoder and a linear head, trained on
[MELD](https://affective-meld.github.io/), whose utterances come from the TV
series *Friends*. Conversational speech is hard: expect the majority classes
(`neutral`, `joy`) far more often than the rare ones.
