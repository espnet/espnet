import pandas as pd
import argparse
import os

try:
    from datasets import load_dataset
except(e):
    print("Error importing datasets library: " + e)

common_voice_split_map = {'train': 'validated', 'validation': 'dev', 'test': 'test'}

parser = argparse.ArgumentParser(description='Download and format FLEURS dataset')
parser.add_argument('--lang', default="all", type=str, help='language to download data for (default: all languages)')
parser.add_argument('--nlsyms_txt', default="nlsyms_txt.txt", type=str, help='a file of iso codes for the lm to ignore')

args = parser.parse_args()

fleurs_asr = load_dataset("fleurs", args.lang)
lang_iso_map = fleurs_asr["train"].features["lang_id"].names

def add_lang_ids(sample):
    lang_iso = lang_iso_map[sample['lang_id']]
    padded_sentence = f"[{lang_iso}] {sample['transcription']}"
    sample['transcription'] = padded_sentence

    return sample

def create_csv(split):
    if args.lang == 'all':
        fleurs_asr[split] = fleurs_asr[split].map(add_lang_ids) # add lang ids if we are doing multilingual processing

    paths = fleurs_asr[split]['path']
    transcriptions = fleurs_asr[split]['transcription']
    ids = fleurs_asr[split]['id']
    langs = fleurs_asr[split]['language']
    pad = [None] * len(fleurs_asr[split]['id']) # these fields aren't used in the common voice recipe

    df = pd.DataFrame(data={'client_id': ids, 'path': paths,'sentence': transcriptions, 'upvotes': pad, 'downvotes': pad, 'age': pad, 'gender': pad, 'accent':langs})
    name = common_voice_split_map[split]
    if not os.path.exists(args.lang):
        os.makedirs(args.lang)

    df.to_csv(f'{args.lang}/{name}.tsv', index=False, sep='\t')

create_csv('train')
create_csv('validation')
create_csv('test')

if args.lang == 'all':
    with open(args.nlsyms_txt, 'w') as fp:
        for iso in lang_iso_map:
            fp.write(iso + '\n')
    print('Saved non-linguistic symbols to ' + args.nlsyms_txt)