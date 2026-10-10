F5-TTS is a zero-shot text-to-speech model: it speaks the text you give it in the voice of a short recording you provide.

**Inputs**

- **Text to synthesize**: the sentence to speak, in a language the model was trained on; punctuation is read as pauses.
- **Reference speech**: the voice to clone. A clean recording of one speaker, ideally 3 to 10 seconds, any sample rate (it is resampled). The output copies this voice, its pace and its recording conditions.
- **Reference transcript**: exactly what is said in the reference recording, required. The model reads the reference through its transcript and sets the length of the output from the reference's speech rate, so a wrong or missing transcript gives garbled or badly timed speech.

**Output**: the synthesized speech at 24 kHz.

When the demo ships an example, the inputs open filled with it: press **Synthesize** to hear it, or replace them with your own; the example row at the bottom restores them. The first call after a pause takes longer while the model loads.
