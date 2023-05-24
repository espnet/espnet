# Converting audio file formats using format_wav_scp.py

The [format_wav_scp.py](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/asr1/pyscripts/audio/format_wav_scp.py) is an utility to convert the audio format of the files specified `wav.scp`
and the [format_wav_scp.sh](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/scripts/audio/format_wav_scp.sh) is a shell script wrapping `format_wav_scp.py`.
In the typical case, in the stage3 of the [template recipe](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE),
`format_wav_scp.sh` is used to convert the audio file format of your original corpus to the audio format which you actually want to feed to the DNN model.

`format_wav_scp.py` and `format_wav_scp.sh` has same function of generation `wav.scp` from `wav.scp`, but　`format_wav_scp.sh` is different in that　it has the capability of parallel processing.

```
wav.scp -> [format_wav_scp.py] -> wav.scp

wav.scp -> [format_wav_scp.sh] -> wav.scp
```

Note that `format_wav_scp.py` dumps audio files with linear PCM with `sint16` regardless the input audio format.

## Quick usage


At the first, you need to prepare a text file named as `wav.scp`:

```
ID_a /some_where/a.wav
ID_b /some_where2/b.wav
...
```

`ID_a`and `ID_b` are the IDs which you can name arbitrarily to specify audio files. Note that **we don't assume any directory stuctures for the audio files**.


```sh
# Please change directory before using our shell scripts
cd egs2/some_corpus/some_task

cmd=utils/run.pl
nj=10  # Number of parallel jobs
audio_format=flac  # The audio codec of output files
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
i.e., the audio codec (`wav` of `linear PCM`, `flac`, `mp3`, `DSD`, `u-law`, `a-law`or etc.), the sampling frequency (`48khz`, `44.1khz`, `16khz`, `8khz`, or etc.),
the bit depth (`uint8`, `sint16`, `sint32`, `float20`, `float32` or etc.),
the number of channels (`monaural`, `stereo`, or more than 2ch), the byter order(`little endian` or `big endian`).

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
for data loading, and, thus the supported audio codecs depend on [libsndfile](http://www.mega-nerd.com/libsndfile/).

You can check the supported audio codecs of `soundfile` with the following command:

```python
import soundfile
print(soundfile.available_formats())
```

Note that the `wav.scp` of Kaldi originally requires that the audio format is wav with pcm_s16le type,
but **`wav.scp` of ESPnet2 can handle all audio formats supported by soundfile**. e.g. You can use `flac` format in `wav.scp` for the input/output of `format_wav_scp.py`.

Depending on the situation, you may choose one of the following codecs:

|  Codec  |  Compression | Maximum channnels | Maximum sampling frequency|Note|
| ---- | ---- | ---- | ---- | ---- |
|  wav (Microsoft wav with linear pcm) | No |  1024  | - | |
|  flac  |  Lossless  | 8 | 192khz ||
| mp3 | Lossy | 2 | 48khz | The patent of MP3 has expired |
| ogg (Vorbis) | Lossy | 255 | 192khz | Segmentation fault happens |


By default, we select `flac` because `flac` can convert linear pcm files with compression rate of ~55 % without data loss.
`flac` is helpful to reduce the IO load, especially, when training with a large amount of corpus.
If you would like to change it to the other format, please use `--audio_format` option for `run.sh`.

```sh
cd egs2/some_corpus/some_task
./run.sh --audio_format mp3
```

Note that if the audio files in your corpus are disributed with lossy audio codec, such as `MP3`,
it's better to keep the file format to avoid the duplication of the full corpus with the uncompressed format.　 **If the input audio format type is exactly same as the output format, `format_wav_scp.py` avoid the gengeration of the output files and reuse the input files**.

## Use case


### Case1: Extract segmentations with long recoding

Create `wav.scp` and `segments` with the format of `The format is <utterance_id> <wav_id> <start_time> <end_time>` (second unit).

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
  - `<num>c|` specifies `<num>` of output channels
  - `|c<out-channel>=c<in-channel>` assigns `<in-channel>`th channel of input stream into `<out-channel>`th channel of output stream
- Caution: `-map_channel` option is deprecated and will be removed.

### Case3: Convert NIST Sphere files to wav

`sph2pipe` is required. Create `wav.scp` as following:

```
ID_a sph2pipe -f wav -p -c 1 ID_a.sph |
ID_b sph2pipe -f wav -p -c 1 ID_b.sph |
...
```



### Case4: Using a mechanism for multi channels inputs

If you are going to generate multi channels audio file from monaural audio files,
create the following wav.scp:

```
ID_a a1.wav a2.wav
...
```

and run the following commands:

```sh
./scripts/audio/format_wav_scp.sh --multi_columns_input true wav.scp output_dir
```

Conversely, if you and going to monaural audio files from multi channels audio files


```sh
./scripts/audio/format_wav_scp.sh --multi_columns_output true wav.scp output_dir
```

Then, you can get `wav.scp` like the following file:

```
ID_a output_dir/IDa-CH0.wav output_dir/ID_a-CH1.wav
...
