import pandas as pd
import argparse
import traceback

try:
    from datasets import load_dataset
except:
    traceback.print_exc()
    print("Error importing datasets library")
    print("datasets can be installed via espnet/tools/installers/install_datasets")

common_voice_split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}

parser = argparse.ArgumentParser(description='Download and format FLEURS dataset')
parser.add_argument('--lang', default="all", type=str, help='language to download data for (default: all languages)')
parser.add_argument('--nlsyms_txt', default="nlsyms_txt.txt", type=str, help='a file of iso codes for the lm to ignore')

args = parser.parse_args()

'''
We use the fleurs portion of "google/xtreme_s" instead of "google/fleurs".
google/fleurs data does not include the full path to the downloaded audio clips, making it harder to process.
'''
fleurs_asr = load_dataset("google/xtreme_s",f"fleurs.{ args.lang}", cache_dir='downloads/cache/')
lang_iso_map = fleurs_asr["train"].features["lang_id"].names

def add_lang_ids(sample):
    lang_iso = lang_iso_map[sample['lang_id']]
    padded_sentence = f"[{lang_iso}] {sample['transcription']}"
    sample['transcription'] = padded_sentence

    return sample

'''
kaldi data validation fails on certain white space characters, those are replaced here
see https://apps.timwhitlock.info/unicode/inspect/hex/2000-206F for details on replaced chars
'''
def replace_bad_spaces(sample):
    sentence = sample['transcription']
    
    sentence = sentence.strip()
    for i in range(8192, 8208):
        sentence = sentence.replace(chr(i), " ")
    for i in range(8232, 8240):
        sentence = sentence.replace(chr(i), " ")
    sentence = sentence.replace(chr(160), " ")

    sample['transcription'] = sentence

    return sample

def create_csv(split):
    if args.lang == 'all':
        fleurs_asr[split] = fleurs_asr[split].map(add_lang_ids) # add lang ids if we are doing multilingual processing
    fleurs_asr[split] = fleurs_asr[split].map(replace_bad_spaces)
    fleurs_asr[split] = fleurs_asr[split].filter(lambda example: example['id'] != 10) # sample 10 has some weird whitespacing
    paths = fleurs_asr[split]['path']
    transcriptions = fleurs_asr[split]['transcription']
    ids = fleurs_asr[split]['id']
    langs = fleurs_asr[split]['language']
    pad = [None] * len(fleurs_asr[split]['id']) # these fields aren't used in the common voice recipe

    df = pd.DataFrame(data={'client_id': ids, 'path': paths,'sentence': transcriptions, 'upvotes': pad, 'downvotes': pad, 'age': pad, 'gender': pad, 'accent':langs})
    name = common_voice_split_map[split]

    df.to_csv(f'downloads/{args.lang}/{name}.tsv', index=False, sep='\t')

create_csv('train')
create_csv('validation')
create_csv('test')

if args.lang == 'all':
    with open(args.nlsyms_txt, 'w') as fp:
        for iso in lang_iso_map:
            fp.write(iso + '\n')
    print('Saved non-linguistic symbols to ' + args.nlsyms_txt)
