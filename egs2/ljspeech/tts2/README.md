# LJSPEECH RECIPE

This is the recipe of English single female speaker TTS model with [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) corpus.

### Discrete FastSpeech

To train the discrete fastspeech model, you need to first prepare the duration information.

It can be either obtained from force-aligner or attention-based auto-regressive model (e.g., Tacotron2).
Please refer to [Alignment from Tacotron2](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/tts1#fastspeech-training) and [Montreal Forced Aligner](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/tts1#new-mfa-aligments-generation).


### Discrete Speech Challenge Baseline

<table class="table">
  <thread>
    <tr>
      <th scope="col">Model</th>
      <th scope="col">MCD</th>
      <th scope="col">Log F0 RMSE</th>
      <th scope="col">WER</th>
      <th scope="col">UTMOS</th>
    </tr>
  </thread>
  <tbody>
    <tr>
      <th scope="col">HuBERT-base-layer6</th>
      <th scope="col">7.19</th>
      <th scope="col">0.26</th>
      <th scope="col">8.1</th>
      <th scope="col">3.73</th>
    </tr>
  </tbody>
</table>
