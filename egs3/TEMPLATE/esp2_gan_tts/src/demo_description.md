This is a multi-speaker GAN-TTS model (VITS, say): it speaks the text you give it in the voice of a short reference recording.

**Inputs**

- **Text to synthesize**: the sentence to speak, in the language the model was trained on. It is cleaned and converted to phonemes the same way the training transcripts were.
- **Reference speech**: the voice to use. A clean recording of one speaker, a few seconds long, any sample rate (it is resampled). The demo turns it into the speaker embedding the model was trained on, so the output matches that voice through the embedding rather than copying the recording.

**Output**: the synthesized speech, mono, at the model's sample rate.

The first call after a pause takes longer while the model and the embedding extractor load.
