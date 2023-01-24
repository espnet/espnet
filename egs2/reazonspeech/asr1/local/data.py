import os
from datasets import load_dataset
from reazonspeech.text import normalize

def save_kaldi_format(outdir, ds):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'text'), 'w') as fp_text, \
         open(os.path.join(outdir, 'wav.scp'), 'w') as fp_wav, \
         open(os.path.join(outdir, 'utt2spk'), 'w') as fp_utt2spk, \
         open(os.path.join(outdir, 'spk2utt'), 'w') as fp_spk2utt:

        for item in ds.sort("name"):
            path = item["audio"]["path"]

            # '11時のニュースです。' -> '１１時のニュースです'
            text = normalize(item["transcription"])

            # '000/e7fb3323c280c.flac' -> '000e7fb3323c280c'
            name = os.path.splitext(item["name"].replace('/', ''))[0]
            uttid = 'uttid%s' % name
            spkid = 'spkid%s' % name
            print(uttid, text, file=fp_text)
            print(uttid, path, file=fp_wav)
            print(uttid, spkid, file=fp_utt2spk)
            print(spkid, uttid, file=fp_spk2utt)

def main():
    ds = load_dataset("reazon-research/reazonspeech", "all", cache_dir="downloads/cache")["train"]
    save_kaldi_format('data/dev', ds.select(range(1000)))
    save_kaldi_format('data/test', ds.select(range(1000, 2000)))
    save_kaldi_format('data/train', ds.select(range(2000, ds.num_rows)))

if __name__ == '__main__':
    main()
