### Pre-processing

* To get mouth ROIs

Run mouth cropping script to save grayscale mouth ROIs. We assume you save cropped mouths to *`$TCN_LIPREADING_ROOT/datasets/visual_data/`*. You can choose `--testset-only` to produce testing set.

```Shell
python crop_mouth_from_video.py --video-direc <LRW-DIREC> \
                                --landmark-direc <LANDMARK-DIREC> \
                                --save-direc <MOUTH-ROIS-DIRECTORY> \
                                --convert-gray \
                                --testset-only
```

* To get audio waveforms

Run format conversion script to extract audio waveforms (.npz) from raw videos. We assume you save audio waveforms to *`$TCN_LIPREADING_ROOT/datasets/audio_data/`*. You can choose `--testset-only` to produce testing set.

```Shell
python extract_audio_from_video.py --video-direc <LRW-DIREC> \
                                   --save-direc <AUDIO-WAVEFORMS-DIRECTORY> \
                                   --testset-only
```
