# OGI Kids spontaneous RECIPE

This is the recipe of the children speech recognition model with [OGI Kids spontaneous dataset](https://catalog.ldc.upenn.edu/LDC2007S18).

Before running the recipe, please download from [LDC](https://catalog.ldc.upenn.edu/LDC2007S18).
Then, edit 'OGI_KIDS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
OGI_KIDS=/path/to/ogi_kids 

```

## WER

|dataset|Err|
|---|---|
|whisper-base.en_zs/dev|28.3|
|whisper-base.en_zs/test|37.3|
|whisper-small.en_zs/dev|21.6|
|whisper-small.en_zs/test|25.3|
|whisper-base.en_4kft/dev|17.3|
|whisper-base.en_4kft/test|16.9|
|whisper-small.en_4kft/dev|10.5|
|whisper-small.en_4kft/test|11.5|

## References

[1] Shobaki, Khaldoun, John-Paul Hosom, and Ronald Cole. "The OGI kids’ speech corpus and recognizers." Proc. of ICSLP. 2000.
