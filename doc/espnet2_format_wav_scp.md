# Converting audio file formats using format_wav_scp.py

The [format_wav_scp.py](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/asr1/pyscripts/audio/format_wav_scp.py) is an utility to convert the audio format of the files specified `wav.scp`
and the [format_wav_scp.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/scripts/audio/format_wav_scp.sh) is a shell script wrapping `format_wav_scp.py`.
In the typical case, in the stage3 of [the template recipe](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE),
`format_wav_scp.sh` is used to convert the audio file format of your original corpus to the audio format which you actually want to feed to the DNN model.

Note that `format_wav_scp.py` dumps files with linear PCM with `sint16le` regardless the input audio format.

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

# Then, you can find output_dir/wav.scp
```

See also:

- About `wav.scp`: https://github.com/espnet/data_example
- About `cmd`:  [Using job scheduling system](parallelization.md)


## Why is audio file formatting necessary?

The audio data included in the corpus obtained from the source website are distributed in various audio file formats,
i.e., the audio codec (`wav` with `linear PCM`, `u-law`, `a-law`, `flac`, `mp3`, or etc.), the sampling frequency (`48khz`, `44.1khz`, `16khz`, `8khz`, or etc.),
the data precision and the type (`uint8`, `sint16`, `sint32`, `float20`, `float32` or etc.),
the number of channels (`monaural`, `stereo`, or more than 2ch), little endian / big endian.

When you try to develop a new recipe with a corpus that is not yet prepared in our recipes,
of course, you can also try to use the audio data as they are without any formatting.
However,
in a typical case, the configuration of our DNN model may assume the specific audio format,
especially regarding the sampling frequency and the data precision.
If you　are conservative with your new recipe,
we recommend converting them to the original recipe's audio format.
For example, `16khz` and `sint16` audio is typically used in our ASR recipes.


## The audio file formats supported in ESPnet2

ESPnet adopts [python soundifile](https://github.com/bastibe/python-soundfile)
for data loading, and, thus the supported audio codec depend on [libsndfile](http://www.mega-nerd.com/libsndfile/).

You can check the supported audio codecs of `soundfile` with the following command:

```python
import soundfile
print(soundfile.available_formats())
```

Depending on the situation, you may choose one of the following codecs:

|  Codec  |  Compression | Maximum number of channnels |Note|
| ---- | ---- | ---- | ---- |
|  wav (Microsoft wav with linear pcm) | No |  1024  | |
|  flac  |  Lossless  | 8 | |
| mp3 | Lossy | 2 | The patent of MP3 has expired |
| ogg (Vorbis) | Lossy | ? | Segmentation fault sometimes happens |


By default, we select `flac` because `flac` can convert linear pcm files with compression rate of ~55 % without data loss.
If you would like to change it to the other format, please use `--audio_format` option for `run.sh`.

```sh
cd egs2/some_corpus/some_task
./run.sh --audio_format mp3
```

Note that if the audio files in your corpus are disributed with lossy audio codec, such as `MP3`,
it's better to keep the file format to avoid the duplication with massive audio format.

## Use case


### Case1: Extract segmentations with long recoding

Create `wav.scp` and `segments` with the format of `The format is <utterance_id> <wav_id> <start_time> <end_time>`.
Note that the time is in second unit.

`wav.scp`:

```
record_a a.wav
...
```

`segments`:

```
segment_a record_a 0.98 11.56
segment_a record_a 12.34 15.43
...
```



Then, you can extract the segments with:


```sh
./scripts/audio/format_wav_scp.sh --segments segments wav.scp output_dir
```

### Case2: Extract audio data from video codec / Use non supported format by soundfile

`ffmpeg` is required. Create `wav.scp` as following:

```
ID_a ffmpeg -i "ID_a.mp4" -f wav -af pan="1c|c0=c0" -acodec pcm_s16le - |
ID_b ffmpeg -i "ID_b.mp4" -f wav -af pan="1c|c0=c0" -acodec pcm_s16le - |
...
```

- Note: `-af pan` is [pan filter](https://ffmpeg.org/ffmpeg-filters.html#pan-1).
  - `<num>c` specifies `<num>` of output channels
  - `c<out-channel>=c<in-channel>` assigns `<in-channel>`th channel
- Caution: `-map_channel` option is deprecated and will be removed.
