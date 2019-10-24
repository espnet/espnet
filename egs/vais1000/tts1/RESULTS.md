# v.0.5.3: transformer.v1 1024 pt window / 256 pt shift / GL 64 iters

This is a demonstration of transformer TTS with VAIS1000 dataset.  You should have significantly better model with just adding more data. The VAIS1000 dataset only contain 1000 utterance and total audio length was about one hour. At least 7 hours is needed to get natural, clean and intelligible speech.

Check the sample on https://drive.google.com/open?id=1MOPHl7aaYJkuIQYEjvCq6VoYFQWKksU6

As you can see, even with considerably little data, we can get some intelligible speech.

You can later train FastSpeech with Transformer trained model.
