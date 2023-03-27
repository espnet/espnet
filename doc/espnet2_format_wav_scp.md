# Converting audio file formats using format_wav_scp.py

The [format_wav_scp.py](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/asr1/pyscripts/audio/format_wav_scp.py) is an utility to convert the audio format of the files specified `wav.scp` 
and the [format_wav_scp.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/scripts/audio/format_wav_scp.sh) is a shell script wrapping `format_wav_scp.py`. 
In the typical case, in the stage3 of [the template recipe](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE), 
`format_wav_scp.sh` is used to convert the audio file format of your original corpus to the audio format which you actually want to feed to the DNN model. 

## Quick usage

```sh
# Please change directory before using our shell scripts
cd egs2/some_corpus/some_task

cmd=utils/run.pl
nj=10  # Number of parallel jobs
audio_fomrat=flac  # The audio codec of output files
fs=16k  # The sampling frequency of output files
ref_channels=0  # If the input data has multiple channels and you want to use only a single channel in the file (please spicify the channel with 0-based number)
./scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${cmd}" --audio_format "${audio_format}" --fs "${fs}" --ref_channels "${ref_channels}" somewhere/wav.scp output_dir

# The, you can find output_dir/wav.scp
```

See also:

- About `wav.scp`: https://github.com/espnet/data_example
- About `cmd`:  [Using job scheduling system](parallelization.md)


## Why is audio file formatting necessary?

The audio data included in the corpus obtained from the source website are distributed in various audio file formats, 
i.e., the audio codec (`wav`, `flac`, `mp3`, or etc.), the sampling frequency (`48khz`, `44.1khz`, `16khz`, `8khz`, or etc.), 
the data precision and the type (`uint8`, `int16`, `int32`, `float20`, `float32`, or etc.), 
the number of channels (`monaural`, `stereo`, or more than 2ch). 

When you try to develop a new recipe with a corpus that is not yet prepared in our recipes, 
of course, you can also try to use the audio data as they are without any formatting. 
However, 
in a typical case, the configuration of our DNN model may assume the specific audio format,
especially regarding the sampling frequency and the data precision. 
If you　are conservative with your new recipe,
we recommend converting them to the original recipe's audio format. 
For example, 16khz and int16 audio files are typically used in our ASR recipes.


## The audio file formats supported in ESPnet2

ESPnets adopts [python soundifile](https://github.com/bastibe/python-soundfile) 
for data loading, and, thus the supported audio codec depend on [libsndfile](http://www.mega-nerd.com/libsndfile/).

You can check the supported audio codecs of `soundfile` with the following command:

```python
import soundfile
print(soundfile.available_formats())
```

Depending on the situation, you may choose one of the following three formats:

|  codec  |  compression | Some notes|
| ---- | ---- | ---- |
|  wav (microsoft wav with linear pcm)  |  No  | |
|  flac  |  Lossless  | Maximum channel number is 8ch |
| mp3 | Lossy | The patent of MP3 has expired |


By default, we select `flac` because `flac` can convert linear pcm files with ~55 % compression rate the audio without data loss.
If you would like to change it to the other format, please use `--audio_format` option for `run.sh`. 

```sh
cd egs2/some_corpus/some_task
./run.sh --audio_format mp3
```
